use std::{net::TcpListener, sync::Arc, time::Duration};

use arrow_schema::{DataType, Field, Schema};
use rust_memex::DEFAULT_TABLE_NAME;
use tokio::{
    process::Command,
    time::{sleep, timeout},
};

fn rust_memex_bin() -> &'static str {
    env!("CARGO_BIN_EXE_rust-memex")
}

fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind free port");
    listener.local_addr().expect("local addr").port()
}

fn schema(include_source_hash: bool) -> Schema {
    let mut fields = vec![
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
    ];

    if include_source_hash {
        fields.push(Field::new("source_hash", DataType::Utf8, true));
    }

    Schema::new(fields)
}

async fn create_table(db_path: &str, include_source_hash: bool) {
    lancedb::connect(db_path)
        .execute()
        .await
        .expect("connect lancedb")
        .create_empty_table(DEFAULT_TABLE_NAME, Arc::new(schema(include_source_hash)))
        .execute()
        .await
        .expect("create table");
}

async fn assert_check_only_sees_migration_needed(db_path: &str) {
    let output = Command::new(rust_memex_bin())
        .args(["--db-path", db_path, "migrate-schema", "--check-only"])
        .output()
        .await
        .expect("run migrate-schema --check-only");
    assert!(
        !output.status.success(),
        "check-only should fail on pre-v4 schema: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

async fn stderr_from_timed_daemon(args: &[String], runtime: Duration) -> String {
    let mut child = Command::new(rust_memex_bin())
        .args(args)
        .stderr(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .expect("spawn daemon");

    sleep(runtime).await;
    let _ = child.kill().await;
    let output = child
        .wait_with_output()
        .await
        .expect("collect daemon output");
    String::from_utf8_lossy(&output.stderr).to_string()
}

#[tokio::test]
async fn daemon_startup_refuses_pre_v4_schema_without_auto_migrate() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let db_path = tmp.path().join("lancedb").to_string_lossy().to_string();
    create_table(&db_path, false).await;
    assert_check_only_sees_migration_needed(&db_path).await;

    let output = timeout(
        Duration::from_secs(5),
        Command::new(rust_memex_bin())
            .args([
                "--db-path",
                db_path.as_str(),
                "sse",
                "--port",
                &free_port().to_string(),
            ])
            .output(),
    )
    .await
    .expect("daemon should fail before binding")
    .expect("run daemon");

    assert!(
        !output.status.success(),
        "daemon should refuse pre-v4 schema without --auto-migrate"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("ERROR: binary requires schema v4, table is v3-pre."),
        "unexpected stderr: {stderr}"
    );
    assert!(
        stderr.contains("Run 'rust-memex migrate-schema --db-path"),
        "unexpected stderr: {stderr}"
    );
    assert!(
        stderr.contains("Or pass --auto-migrate to migrate at startup."),
        "unexpected stderr: {stderr}"
    );
}

#[tokio::test]
async fn daemon_startup_auto_migrates_pre_v4_schema() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let db_path = tmp.path().join("lancedb").to_string_lossy().to_string();
    create_table(&db_path, false).await;
    assert_check_only_sees_migration_needed(&db_path).await;

    let args = vec![
        "--db-path".to_string(),
        db_path.clone(),
        "--auto-migrate".to_string(),
        "sse".to_string(),
        "--port".to_string(),
        free_port().to_string(),
    ];
    let stderr = stderr_from_timed_daemon(&args, Duration::from_secs(2)).await;
    assert!(
        stderr.contains("migrating schema v3-pre -> v4 at startup"),
        "unexpected stderr: {stderr}"
    );

    let table = lancedb::connect(&db_path)
        .execute()
        .await
        .expect("connect")
        .open_table(DEFAULT_TABLE_NAME)
        .execute()
        .await
        .expect("open migrated table");
    table.checkout_latest().await.expect("checkout latest");
    let schema = table.schema().await.expect("schema");
    assert!(
        schema.field_with_name("source_hash").is_ok(),
        "source_hash should exist after --auto-migrate"
    );
}

#[tokio::test]
async fn daemon_startup_on_v4_schema_has_no_schema_guard_noise() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let db_path = tmp.path().join("lancedb").to_string_lossy().to_string();
    create_table(&db_path, true).await;

    let args = vec![
        "--db-path".to_string(),
        db_path,
        "sse".to_string(),
        "--port".to_string(),
        free_port().to_string(),
    ];
    let stderr = stderr_from_timed_daemon(&args, Duration::from_secs(2)).await;
    assert!(
        !stderr.contains("binary requires schema"),
        "unexpected stderr: {stderr}"
    );
    assert!(
        !stderr.contains("migrating schema"),
        "unexpected stderr: {stderr}"
    );
}
