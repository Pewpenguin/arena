use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "arena", version, about = "Arena")]
pub struct Cli {
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    pub verbose: u8,

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
