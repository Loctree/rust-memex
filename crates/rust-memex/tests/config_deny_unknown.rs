use std::{fs, process::Command};

use tempfile::TempDir;

fn write_config(tmp: &TempDir, name: &str, contents: &str) -> std::path::PathBuf {
    let path = tmp.path().join(name);
    fs::write(&path, contents).expect("write config");
    path
}

fn run_health_with_config(
    config_path: &std::path::Path,
    home: &std::path::Path,
) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_rust-memex"))
        .env("HOME", home)
        .arg("--config")
        .arg(config_path)
        .args(["health", "--quick", "--json"])
        .output()
        .expect("run rust-memex health")
}

#[test]
fn db_path_inside_embeddings_section_is_rejected() {
    let tmp = TempDir::new().expect("tmp");
    let config = write_config(
        &tmp,
        "misnested-db-path.toml",
        r#"
[embeddings]
required_dimension = 4096
db_path = "/tmp/should-not-be-read"
"#,
    );

    let output = run_health_with_config(&config, tmp.path());

    assert!(
        !output.status.success(),
        "misnested db_path must fail loudly"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("unknown field"), "{stderr}");
    assert!(stderr.contains("db_path"), "{stderr}");
}

#[test]
fn missing_top_level_db_path_uses_default_and_warns() {
    let tmp = TempDir::new().expect("tmp");
    let config = write_config(
        &tmp,
        "default-db-path.toml",
        r#"
[embeddings]
required_dimension = 4096
max_batch_chars = 32000
max_batch_items = 16
"#,
    );

    let output = run_health_with_config(&config, tmp.path());

    assert!(output.status.success(), "health failed: {output:?}");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("WARN: config db_path not specified"),
        "{stderr}"
    );
    assert!(
        stderr.contains(".rmcp-servers/rust-memex/lancedb"),
        "{stderr}"
    );
    assert!(
        stderr.contains("before any section such as [embeddings]"),
        "{stderr}"
    );
}

#[test]
fn top_level_db_path_is_used_without_default_warning() {
    let tmp = TempDir::new().expect("tmp");
    let db_path = tmp.path().join("custom-lancedb");
    let config = write_config(
        &tmp,
        "top-level-db-path.toml",
        &format!(
            r#"
db_path = "{}"

[embeddings]
required_dimension = 4096
max_batch_chars = 32000
max_batch_items = 16
"#,
            db_path.display()
        ),
    );

    let output = run_health_with_config(&config, tmp.path());

    assert!(output.status.success(), "health failed: {output:?}");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("WARN: config db_path not specified"),
        "{stderr}"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains(&db_path.display().to_string()), "{stdout}");
}
