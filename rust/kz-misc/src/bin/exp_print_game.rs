// use board_game::arimaa_engine_step::full_move::convert_actions_to_move_string;
use board_game::board::{Board, Player};
use board_game::games::arimaa::ArimaaBoard;

use kz_core::mapping::arimaa::ArimaaSplitMapper;
use kz_core::mapping::{BoardMapper, PolicyMapper};

use crate::indices::INDICES;

fn main() {
    let mapper = ArimaaSplitMapper;

    let mut all_indices = start_indices(&mapper);
    all_indices.extend_from_slice(INDICES);

    let mut board = ArimaaBoard::default();

    let mut prev_board = board.clone();
    let mut actions = vec![];

    for &mv_index in &all_indices {
        let mv = mapper.index_to_move(&board, mv_index).unwrap();
        actions.push(mv);
        board.play(mv).unwrap();

        if board.next_player() != prev_board.next_player() {
            // TODO uncomment this
            // let line = convert_actions_to_move_string(prev_board.state().clone(), &actions);

            let player_char = match prev_board.next_player() {
                Player::A => 'g',
                Player::B => 's',
            };

            // TODO uncomment this
            // println!("{}{} {}", prev_board.state().move_number(), player_char, line);

            prev_board = board.clone();
            actions.clear();
        }
    }
}

fn start_indices(mapper: &impl BoardMapper<ArimaaBoard>) -> Vec<usize> {
    // use board_game::arimaa_engine_step::convert_char_to_piece;
    use board_game::arimaa_engine_step::Action;

    let pieces = "rhcmechrrrrddrrrrrrddrrrrhcemchr";
    let mut board = ArimaaBoard::default();
    let mut indices = vec![];

    for p in pieces.chars() {
        // TODO uncomment
        // let mv = Action::Place(convert_char_to_piece(p).unwrap().0);
        // indices.push(mapper.move_to_index(&board, mv));

        // board.play(mv)
    }

    indices
}

#[rustfmt::skip]
mod indices {
    pub const INDICES: &[usize] = &[56, 48, 168, 112, 60, 123, 57, 62, 119, 113, 63, 242, 61, 0, 59, 58, 179, 62, 252, 65, 69, 59, 59, 241, 0, 54, 57, 250, 70, 54, 0, 128, 132, 253, 254, 67, 70, 121, 260, 133, 243, 53, 45, 250, 61, 261, 51, 131, 61, 165, 230, 109, 59, 179, 256, 117, 57, 252, 177, 107, 60, 114, 66, 56, 240, 123, 119, 57, 58, 48, 177, 50, 175, 234, 257, 55, 232, 259, 257, 246, 117, 185, 66, 132, 128, 108, 123, 252, 262, 123, 56, 69, 39, 242, 121, 65, 185, 233, 69, 229, 45, 37, 250, 236, 260, 67, 171, 130, 57, 238, 131, 46, 65, 60, 64, 65, 156, 221, 260, 123, 38, 58, 108, 252, 127, 253, 248, 39, 232, 237, 61, 103, 249, 259, 45, 112, 124, 103, 241, 101, 0, 121, 120, 249, 51, 54, 117, 116, 66, 53, 250, 235, 232, 238, 54, 104, 169, 39, 44, 87, 31, 48, 124, 58, 0, 127, 15, 23, 121, 50, 115, 251, 56, 55, 256, 143, 61, 104, 63, 244, 64, 151, 181, 48, 114, 122, 115, 233, 253, 248, 122, 96, 95, 254, 41, 240, 125, 119, 170, 178, 125, 234, 254, 174, 120, 224, 127, 250, 159, 251, 97, 233, 226, 228, 248, 251, 58, 33, 244, 237, 53, 50, 246, 237, 117, 119, 111, 243, 114, 42, 36, 64, 108, 240, 54, 57, 174, 39, 56, 159, 121, 232, 43, 51, 109, 122, 107, 163, 98, 61, 250, 121, 182, 104, 243, 0, 58, 117, 254, 242, 99, 103, 178, 100, 39, 36, 99, 164, 59, 243, 41, 114, 246, 245, 189, 44, 97, 122, 34, 251, 120, 261, 260, 121, 218, 243, 37, 254, 122, 174, 52, 259, 29, 21, 242, 177, 54, 124, 258, 228, 112, 61, 85, 141, 44, 238, 65, 0, 181, 49, 189, 133, 35, 36, 28, 20];
}
