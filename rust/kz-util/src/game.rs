use std::fmt::{Display, Formatter};

#[derive(Debug, Copy, Clone)]
pub enum Game {
    TTT,
    STTT,
    Chess,
    ChessHist { length: usize },
    Ataxx { size: u8 },
    ArimaaSplit,
    Go { size: u8 },
}

impl Game {
    pub fn parse(str: &str) -> Option<Game> {
        match str {
            "ttt" => return Some(Game::TTT),
            "sttt" => return Some(Game::STTT),
            "chess" => return Some(Game::Chess),
            "ataxx" => return Some(Game::Ataxx { size: 7 }),
            "arimaa-split" => return Some(Game::ArimaaSplit),
            _ => {}
        };

        if let Some(size) = str.strip_prefix("ataxx-") {
            let size: u8 = size.parse().ok()?;
            return Some(Game::Ataxx { size });
        }
        if let Some(length) = str.strip_prefix("chess-hist-") {
            let length: usize = length.parse().ok()?;
            return Some(Game::ChessHist { length });
        }
        if let Some(size) = str.strip_prefix("go-") {
            let size: u8 = size.parse().ok()?;
            return Some(Game::Go { size });
        }

        None
    }
}

impl Display for Game {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Game::TTT => write!(f, "ttt"),
            Game::STTT => write!(f, "sttt"),
            Game::Chess => write!(f, "chess"),
            Game::ChessHist { length } => write!(f, "chess-hist-{}", length),
            Game::Ataxx { size } => write!(f, "ataxx-{}", size),
            Game::ArimaaSplit => write!(f, "arimaa-split"),
            Game::Go { size } => write!(f, "go-{}", size),
        }
    }
}
