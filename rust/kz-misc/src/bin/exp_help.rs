use board_game::board::Board;
use board_game::games::ataxx::{AtaxxBoard, Move};

fn main() {
    let board = AtaxxBoard::from_fen("ooooxxx/xoooxx1/xxoxxx1/xxoxxxx/ooooxxx/oooooox/1ooooox o 1 44").unwrap();
    println!("{}", board.moves_since_last_copy());

    println!("{}", board);
}
