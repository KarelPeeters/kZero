// #![allow(unreachable_code)]
// #![allow(dead_code)]
//
// use board_game::board::Board;
// use board_game::games::chess::ChessBoard;
// use board_game::games::ttt::TTTBoard;
// use board_game::util::coord::Coord3;
// use kz_core::mapping::chess::ChessHistoryMapper;
// use kz_core::mapping::ttt::TTTStdMapper;
// use kz_misc::muzero_debug;
//
// fn main() {
//     unsafe { main_impl_chess() }
// }
//
// unsafe fn main_impl_ttt() {
//     let mapper = TTTStdMapper;
//     let path = r#"C:\Documents\Programming\STTT\AlphaZero\data\muzero\all\unroll-again-nonorm\models_4500_"#;
//
//     let board = TTTBoard::default();
//     // board.play(Coord::from_i(4));
//     // board.play(Coord::from_i(5));
//     // board.play(Coord::from_i(0));
//     // board.play(Coord::from_i(7));
//
//     let moves = [Coord3::from_index(4), Coord3::from_index(1)];
//     muzero_debug::muzero_debug_utility(path, board, mapper, &moves, true, true, Some(10000));
// }
//
// unsafe fn main_impl_chess() {
//     let mapper = ChessHistoryMapper::new(8);
//     let path =
//         r#"C:\Documents\Programming\STTT\AlphaZero\data\loop_mu\chess-hist-8\working\tmp\curr_network\network_32_"#;
//
//     let mut board = ChessBoard::default();
//     board.play(board.parse_move("e2e4").unwrap()).unwrap();
//     board.play(board.parse_move("e7e5").unwrap()).unwrap();
//     board.play(board.parse_move("d1h5").unwrap()).unwrap();
//     board.play(board.parse_move("e8e7").unwrap()).unwrap();
//
//     muzero_debug::muzero_debug_utility(path, board, mapper, &[], false, false, Some(1000));
// }

fn main() {}
