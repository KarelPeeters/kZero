use board_game::board::BoardMoves;
use board_game::games::ataxx::AtaxxBoard;
use internal_iterator::InternalIterator;

use kz_core::mapping::ataxx::AtaxxStdMapper;
use kz_core::mapping::InputMapper;

fn main() {
    // let board = AtaxxBoard::diagonal(7);
    let board = AtaxxBoard::from_fen("x5o/7/2-1-2/7/2-1-2/7/o5x x 0 1").unwrap();
    let mapper = AtaxxStdMapper::new(board.size());

    println!("{}", board);

    board.available_moves().unwrap().for_each(|mv| {
        let index = mapper.move_to_index(mv);
        println!("{mv}: {index}")
    });

    let mut result = vec![];
    mapper.encode_input_full(&mut result, &board);
    println!("{:?}", result);
}
