use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "arena", version, about = "Arena")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    Run {
        #[arg(long)]
        model: String,
        #[arg(long)]
        prompt: String,
    },
    Exec {
        #[arg(long)]
        tasks: PathBuf,
        #[arg(long = "model", required = true)]
        models: Vec<String>,
        #[arg(long)]
        output: Option<PathBuf>,
    },
}
