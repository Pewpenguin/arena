use std::fmt;
use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

use crate::tournament::{DEFAULT_BEST_OF, TournamentFormat};

#[derive(Debug, Parser)]
#[command(name = "arena", version, about = "Arena")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, ValueEnum)]
pub enum ProviderChoice {
    #[default]
    Openai,
    Openrouter,
    Anthropic,
    Gemini,
}

impl fmt::Display for ProviderChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Openai => "openai",
            Self::Openrouter => "openrouter",
            Self::Anthropic => "anthropic",
            Self::Gemini => "gemini",
        };
        f.write_str(name)
    }
}

impl ProviderChoice {
    pub fn as_persisted(self) -> &'static str {
        match self {
            Self::Openai => "openai",
            Self::Openrouter => "openrouter",
            Self::Anthropic => "anthropic",
            Self::Gemini => "gemini",
        }
    }
}

#[derive(Debug, Subcommand)]
pub enum Command {
    Run {
        #[arg(long, value_enum, default_value_t = ProviderChoice::Openai)]
        provider: ProviderChoice,
        #[arg(long)]
        model: String,
        #[arg(long)]
        prompt: String,
    },
    Exec {
        #[arg(long, value_enum, default_value_t = ProviderChoice::Openai)]
        provider: ProviderChoice,
        #[arg(long)]
        tasks: PathBuf,
        #[arg(long = "model", required = true)]
        models: Vec<String>,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long)]
        judge: Option<String>,
        #[arg(long, value_enum, default_value_t = TournamentFormat::RoundRobin)]
        tournament: TournamentFormat,
        #[arg(long, default_value_t = DEFAULT_BEST_OF)]
        best_of: u32,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    Report {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    Web {
        #[arg(long, default_value_t = 3030)]
        port: u16,
    },
}
