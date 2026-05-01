use anyhow::{Result, anyhow};
use tracing::info;

use crate::{SchemaMigrationReport, SchemaMismatchWriteError, SchemaVersion, StorageManager};

#[derive(Debug, Clone)]
pub enum StartupSchemaGuard {
    UpToDate,
    Migrated(SchemaMigrationReport),
}

pub async fn guard_daemon_startup_schema(
    db_path: &str,
    auto_migrate: bool,
) -> Result<StartupSchemaGuard> {
    let storage = StorageManager::new_lance_only(db_path).await?;

    match storage.require_current_schema_for_writes().await {
        Ok(()) => Ok(StartupSchemaGuard::UpToDate),
        Err(error) => {
            let Some(schema_error) = error.downcast_ref::<SchemaMismatchWriteError>() else {
                return Err(error);
            };
            let db_path = schema_error.db_path().to_string();

            if !auto_migrate {
                return Err(anyhow!(
                    "ERROR: binary requires schema v4, table is v3-pre.\n\
                     Run 'rust-memex migrate-schema --db-path {db_path}' before starting daemon.\n\
                     Or pass --auto-migrate to migrate at startup."
                ));
            }

            let message = "migrating schema v3-pre -> v4 at startup";
            info!("{message}");
            eprintln!("INFO: {message}");
            let report =
                StorageManager::migrate_lance_schema(&db_path, SchemaVersion::current(), false)
                    .await?;
            Ok(StartupSchemaGuard::Migrated(report))
        }
    }
}
