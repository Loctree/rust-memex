use anyhow::Result;
use clap::Parser;

pub mod cli;
use crate::cli::definition::Cli;
use crate::cli::dispatch::run_command;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    run_command(cli).await
}
