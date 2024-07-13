use board_game::board::Board;
use board_game::games::ataxx::{AtaxxBoard, Move};
use board_game::pov::Pov;
use rand::rngs::StdRng;
use rand::SeedableRng;

use kn_cuda_sys::wrapper::handle::Device;
use kn_graph::onnx::load_graph_from_onnx_path;
use kn_graph::optimizer::optimize_graph;
use kz_core::mapping::ataxx::AtaxxStdMapper;
use kz_core::network::cudnn::CudaNetwork;
use kz_core::network::Network;
use kz_core::zero::node::UctWeights;
use kz_core::zero::step::{FpuMode, QMode};
use kz_core::zero::wrapper::ZeroSettings;

// const MOVES: &str = "f2 b2 e1 a2 b7 c1 g2 f6 b6 b3 a6 b3a5 b7b5 a2b4 a6a4 a3 a6 b4d2 f2d1 c2 e1c3 b4 d2b3 d2 g1e2 b5b7 b5 c4 c6 c1e1 c5 c2a2 c4c2 b1 c3c1 e2c3 c4 d2f1 d1f3 e1g3 c7 c2d4 c1e2 c3c1 c2 f2 c3 c5d3 c5 f6d6 e3 c5e4 f2d2 f3d1 c5 d5 e3e1 c1e3 c1 e4f2 d1f3 d1 e3g1 d6f4 e3 e4 b7d6 b7 d7 c4e6 c4 d7e5 c7e7 b7d7 c5b7 d7c5 b7d7 b7 d6f5 d6 a5c7 a5 f4f6 f4 g2g4 e1g2 d7f7 c5d7 c3e1 c1c3 e1c1 e5g6 c1e1 c3c1 e1c3 c1e1 c3c1 g5 c1c3 c5 c3c1 c3 0000 e5";
const MOVES: &str = "f2 b2 e1 a2 b7 c1 g2";

fn main() {
    let path = r#"C:\Documents\Programming\STTT\kZero\data\networks\ataxx_16x128_gen_1364_old.onnx"#;
    let graph = optimize_graph(&load_graph_from_onnx_path(path, false).unwrap(), Default::default());

    let weights = UctWeights::default();

    let batch_size_small = 128;
    let batch_size_large = 2048;

    let settings_small = ZeroSettings::simple(batch_size_small, weights, QMode::wdl(), FpuMode::Relative(0.0));
    let settings_large = ZeroSettings::simple(batch_size_large, weights, QMode::wdl(), FpuMode::Relative(0.0));

    let mapper = AtaxxStdMapper::new(7);
    let mut network_small = CudaNetwork::new(mapper, &graph, batch_size_small, Device::new(0));
    let mut network_large = CudaNetwork::new(mapper, &graph, batch_size_large, Device::new(0));
    let mut rng = StdRng::seed_from_u64(0);

    let mut board = AtaxxBoard::diagonal(7);

    let all_visits = [100];
    let mut all_evals: Vec<Vec<f32>> = vec![vec![]; all_visits.len()];

    for (mi, mv) in MOVES.split(' ').enumerate() {
        println!("{}", board);

        for (vi, visits) in all_visits.into_iter().enumerate() {
            let (eval_rel, batch_size) = if visits == 1 {
                (network_small.evaluate(&board).values, 1)
            } else {
                let (settings, network) = if visits < batch_size_large * 10 {
                    (settings_small, &mut network_small)
                } else {
                    (settings_large, &mut network_large)
                };

                let tree = settings.build_tree(&board, network, &mut rng, |tree| tree.root_visits() >= visits as u64);

                println!("{}", tree.display(usize::MAX, false, usize::MAX, true));

                (tree.values(), settings.batch_size)
            };

            let eval_abs = eval_rel.un_pov(board.next_player());
            all_evals[vi].push(eval_abs.wdl_abs.value().value_a);

            println!(
                "move {} {} visits {} batch_size {}: values {}",
                mi / 2 + 1,
                mv,
                visits,
                batch_size,
                eval_abs
            );
        }

        let mv = Move::from_uai(mv).unwrap();
        board.play(mv).unwrap();
    }

    println!("visits: {:?}", all_visits);
    println!("evals: {:?}", all_evals);
}
