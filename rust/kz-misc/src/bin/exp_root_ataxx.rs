use board_game::board::Board;
use board_game::games::ataxx::AtaxxBoard;
use rand::rngs::StdRng;
use rand::SeedableRng;

use kn_cuda_sys::wrapper::handle::Device;
use kn_graph::onnx::load_graph_from_onnx_path;
use kn_graph::optimizer::optimize_graph;
use kz_core::mapping::ataxx::AtaxxStdMapper;
use kz_core::network::cudnn::CudaNetwork;
use kz_core::zero::node::UctWeights;
use kz_core::zero::step::{FpuMode, QMode};
use kz_core::zero::wrapper::ZeroSettings;

fn main() {
    let path_old = r#"C:\Documents\Programming\STTT\kZero\data\networks\ataxx_16x128_gen_1364.onnx"#;
    let path_gaps = r#"\\192.168.0.10\Documents\Karel A0\loop\ataxx-7\16x128_gaps\training\gen_6732\network.onnx"#;

    let graph_old = optimize_graph(&load_graph_from_onnx_path(path_old, false).unwrap(), Default::default());
    let graph_gaps = optimize_graph(
        &load_graph_from_onnx_path(path_gaps, false).unwrap(),
        Default::default(),
    );

    let size = 7;
    let mapper = AtaxxStdMapper::new(size);
    let batch_size = 1;
    let device = Device::new(0);

    let mut network_old = CudaNetwork::new(mapper, &graph_old, batch_size, device);
    let mut network_gaps = CudaNetwork::new(mapper, &graph_gaps, batch_size, device);

    let visits = 1000;
    let settings = ZeroSettings::new(
        batch_size,
        UctWeights::default(),
        QMode::wdl(),
        FpuMode::Relative(0.0),
        FpuMode::Relative(0.0),
        1.0,
        1.0,
    );
    let mut rng = StdRng::from_entropy();

    // let board = AtaxxBoard::from_fen("x5o/7/2-1-2/7/2-1-2/7/o5x x 0 0").unwrap();
    // let board = AtaxxBoard::from_fen("x3---/-------/-------/-------/-------/-------/o3--- x 0 0").unwrap();
    let board = AtaxxBoard::from_fen("x6/7/-------/-------/-------/7/o6 x 0 0").unwrap();
    println!("{}", board);

    {
        let mut board = board.clone();
        let outcome = loop {
            if let Some(outcome) = board.outcome() {
                break outcome;
            }
            let mv = board.random_available_move(&mut rng).unwrap();
            // let mv = minimax(&board, &SolverHeuristic, 8, &mut rng).best_move.unwrap();
            println!("Playing {}", mv);
            board.play(mv).unwrap();
            println!("{}", board);
        };
        println!("Outcome: {:?}", outcome);
    }

    let tree_old = settings.build_tree(&board, &mut network_old, &mut rng, |tree| tree.root_visits() >= visits);
    println!("Tree old:");
    println!("{}", tree_old.display(1, false, usize::MAX, true));

    let tree_gaps = settings.build_tree(&board, &mut network_gaps, &mut rng, |tree| tree.root_visits() >= visits);
    println!("Tree gaps:");
    println!("{}", tree_gaps.display(1, false, usize::MAX, true,));
}
