// use std::io;
// use std::io::Write;
// use std::path::Path;
// use std::process::{ChildStdin, ChildStdout, Command, Stdio};
//
// use board_game::ai::Bot;
// use board_game::board::BoardDone;
// use board_game::chess::ChessMove;
// use board_game::games::chess::ChessBoard;
//
// fn main() {
//     let mut bot = UciBot::new("~/go/bin/mess.exe").unwrap();
//     let pos = ChessBoard::default();
//
//     let mv = bot.select_move(&pos).unwrap();
//
//     println!("{}", mv);
// }
//
// pub struct UciBot {
//     stdin: ChildStdin,
//     stdout: ChildStdout,
// }
//
// impl UciBot {
//     pub fn new(path: impl AsRef<Path>) -> io::Result<Self> {
//         let path = path.as_ref().as_os_str();
//         let d = Command::new(path)
//             .stdin(Stdio::piped())
//             .stdout(Stdio::piped())
//             .stderr(Stdio::inherit())
//             .spawn()?;
//
//         let mut stdin = d.stdin.unwrap();
//         let stdout = d.stdout.unwrap();
//         assert!(d.stderr.is_none());
//
//         let f = &mut stdin;
//         writeln!(f, "uci")?;
//         writeln!(f, "isready")?;
//
//         Ok(Self { stdin, stdout })
//     }
// }
//
// impl Bot<ChessBoard> for UciBot {
//     fn select_move(&mut self, board: &ChessBoard) -> Result<ChessMove, BoardDone> {
//         let f = &mut self.stdin;
//
//         writeln!(f, "ucinewgame")?;
//         writeln!(f, "position")?;
//
//         todo!()
//     }
// }

fn main() {}
