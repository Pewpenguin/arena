use tracing::Level;

use crate::cli::Cli;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    pub log_level: Level,
}

impl Config {
    pub fn from_cli(cli: &Cli) -> Self {
        let log_level = match cli.verbose {
            0 => Level::INFO,
            1 => Level::DEBUG,
            _ => Level::TRACE,
        };

        Self { log_level }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::Command;

    fn cli_with_verbose(verbose: u8) -> Cli {
        Cli {
            verbose,
            command: Command::Run {
                model: String::new(),
                prompt: String::new(),
            },
        }
    }

    #[test]
    fn maps_verbosity_to_log_level() {
        assert_eq!(
            Config::from_cli(&cli_with_verbose(0)).log_level,
            Level::INFO
        );
        assert_eq!(
            Config::from_cli(&cli_with_verbose(1)).log_level,
            Level::DEBUG
        );
        assert_eq!(
            Config::from_cli(&cli_with_verbose(2)).log_level,
            Level::TRACE
        );
        assert_eq!(
            Config::from_cli(&cli_with_verbose(5)).log_level,
            Level::TRACE
        );
    }
}
