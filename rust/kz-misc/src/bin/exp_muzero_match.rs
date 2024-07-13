// use board_game::board::AltBoard;
// use board_game::games::chess::ChessBoard;
// use board_game::util::board_gen::random_board_with_moves;
// use board_game::util::bot_game;
// use kn_cuda_sys::wrapper::handle::Device;
// use kz_core::mapping::chess::{ChessHistoryMapper, ChessStdMapper};
// use kz_core::mapping::BoardMapper;
// use kz_core::muzero::wrapper::{MuZeroBot, MuZeroSettings};
// use kz_core::network::muzero::{MuZeroFusedGraphs, MuZeroGraphs};
// use kz_core::zero::node::UctWeights;
// use kz_core::zero::step::FpuMode;
// use rand::thread_rng;
//
// fn main() {
//     let path_l = r#"C:\Documents\Programming\STTT\AlphaZero\data\muzero\hist_16\models_1500_"#;
//     let path_r = r#"C:\Documents\Programming\STTT\AlphaZero\data\muzero\uniform_chess_16x128\models_3500_"#;
//
//     let mapper_l = ChessHistoryMapper::new(8);
//     let mapper_r = ChessStdMapper;
//
//     let settings = MuZeroSettings::new(1, UctWeights::default(), false, FpuMode::Relative(0.0), 100);
//     let visits = 100;
//
//     let graph_l = MuZeroGraphs::load(path_l, mapper_l).unwrap().fuse(Default::default());
//     let graph_r = MuZeroGraphs::load(path_r, mapper_r).unwrap().fuse(Default::default());
//
//     let result = bot_game::run(
//         || random_board_with_moves(&ChessBoard::default(), 4, &mut thread_rng()),
//         bot_builder(&graph_l, settings, visits, 1),
//         bot_builder(&graph_r, settings, visits, 1),
//         16,
//         true,
//         |wdl, replay| {
//             println!();
//             println!("{}", replay.to_pgn());
//             println!();
//             println!("Current score: {:?}", wdl);
//         },
//     );
//
//     println!("{:?}", result);
// }
//
// fn bot_builder<B: AltBoard, M: BoardMapper<B>>(
//     graphs: &MuZeroFusedGraphs<B, M>,
//     settings: MuZeroSettings,
//     visits: u64,
//     batch_size: usize,
// ) -> impl Fn() -> MuZeroBot<B, M> + Sync + '_ {
//     move || {
//         let device = Device::new(0);
//         MuZeroBot::new(
//             settings,
//             visits,
//             graphs.mapper,
//             graphs.root_executor(device, batch_size),
//             graphs.expand_executor(device, batch_size),
//         )
//     }
// }

fn main() {}
