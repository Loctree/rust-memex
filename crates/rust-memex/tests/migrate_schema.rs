use std::process::Command;
use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema};
use rust_memex::DEFAULT_TABLE_NAME;

fn rust_memex_bin() -> &'static str {
    env!("CARGO_BIN_EXE_rust-memex")
}

fn pre_v4_schema() -> Schema {
    Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("namespace", DataType::Utf8, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 3),
            false,
        ),
        Field::new("text", DataType::Utf8, true),
        Field::new("metadata", DataType::Utf8, true),
        Field::new("layer", DataType::UInt8, true),
        Field::new("parent_id", DataType::Utf8, true),
        Field::new("children_ids", DataType::Utf8, true),
        Field::new("keywords", DataType::Utf8, true),
        Field::new("content_hash", DataType::Utf8, true),
    ])
}

#[tokio::test]
async fn migrate_schema_adds_source_hash_and_rerun_is_noop() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let db_path = tmp.path().join("lancedb");
    let db_path_str = db_path.to_string_lossy().to_string();

    let db = lancedb::connect(&db_path_str)
        .execute()
        .await
        .expect("connect lancedb");
    db.create_empty_table(DEFAULT_TABLE_NAME, Arc::new(pre_v4_schema()))
        .execute()
        .await
        .expect("create pre-v4 table");

    let check = Command::new(rust_memex_bin())
        .args([
            "--db-path",
            db_path_str.as_str(),
            "migrate-schema",
            "--check-only",
        ])
        .output()
        .expect("run check-only");
    assert!(
        !check.status.success(),
        "check-only should exit non-zero when migration is needed"
    );
    let check_stdout = String::from_utf8_lossy(&check.stdout);
    assert!(
        check_stdout.contains(r#"Migration needed. Missing columns: ["source_hash"]"#),
        "unexpected check-only stdout: {check_stdout}"
    );

    let migrate = Command::new(rust_memex_bin())
        .args(["--db-path", db_path_str.as_str(), "migrate-schema"])
        .output()
        .expect("run migrate-schema");
    assert!(
        migrate.status.success(),
        "migrate-schema failed: stdout={} stderr={}",
        String::from_utf8_lossy(&migrate.stdout),
        String::from_utf8_lossy(&migrate.stderr)
    );
    let migrate_stdout = String::from_utf8_lossy(&migrate.stdout);
    assert!(
        migrate_stdout.contains("Migration complete. Schema is now v4."),
        "unexpected migrate stdout: {migrate_stdout}"
    );

    let backfill = Command::new(rust_memex_bin())
        .args([
            "--db-path",
            db_path_str.as_str(),
            "backfill-hashes",
            "--dry-run",
            "false",
        ])
        .output()
        .expect("run backfill-hashes after schema migration");
    assert!(
        backfill.status.success(),
        "backfill-hashes failed after migration: stdout={} stderr={}",
        String::from_utf8_lossy(&backfill.stdout),
        String::from_utf8_lossy(&backfill.stderr)
    );

    let table = db
        .open_table(DEFAULT_TABLE_NAME)
        .execute()
        .await
        .expect("open migrated table");
    table.checkout_latest().await.expect("checkout latest");
    let schema = table.schema().await.expect("read migrated schema");
    assert!(
        schema.field_with_name("source_hash").is_ok(),
        "source_hash should be present after migration"
    );

    let rerun = Command::new(rust_memex_bin())
        .args(["--db-path", db_path_str.as_str(), "migrate-schema"])
        .output()
        .expect("rerun migrate-schema");
    assert!(
        rerun.status.success(),
        "rerun failed: stdout={} stderr={}",
        String::from_utf8_lossy(&rerun.stdout),
        String::from_utf8_lossy(&rerun.stderr)
    );
    let rerun_stdout = String::from_utf8_lossy(&rerun.stdout);
    assert!(
        rerun_stdout.contains("Schema is up-to-date (target=v4). No migration needed."),
        "unexpected rerun stdout: {rerun_stdout}"
    );
}
