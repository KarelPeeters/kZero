use board_game::board::{Board, Player};
use board_game::games::ataxx::AtaxxBoard;
use rand::rngs::StdRng;
use rand::SeedableRng;

use kn_cuda_eval::Device;
use kn_graph::onnx::load_graph_from_onnx_path;
use kn_graph::optimizer::{optimize_graph, OptimizerSettings};
use kz_core::mapping::ataxx::AtaxxStdMapper;
use kz_core::mapping::BoardMapper;
use kz_core::network::cudnn::CudaNetwork;
use kz_core::zero::node::UctWeights;
use kz_core::zero::step::{FpuMode, QMode};
use kz_core::zero::wrapper::{ZeroBot, ZeroSettings};

fn main() {
    let path_l = "C:/Documents/Programming/STTT/AlphaZero/data/loop/ataxx-6/initial/training/gen_90/network.onnx";
    let path_r = "C:/Documents/Programming/STTT/AlphaZero/data/loop/ataxx-6/small-eps/training/gen_90/network.onnx";

    let batch_size = 1;
    let settings_l = ZeroSettings::simple(batch_size, UctWeights::default(), QMode::wdl(), FpuMode::Relative(0.0));
    let settings_r = ZeroSettings::simple(batch_size, UctWeights::default(), QMode::wdl(), FpuMode::Relative(0.0));
    let visits_l = 1000;
    let visits_r = 10_000;

    println!("{} {:?}", path_l, settings_l);
    println!("vs");
    println!("{} {:?}", path_r, settings_r);

    let size = 6;
    let mapper = AtaxxStdMapper::new(size);

    let mut board = AtaxxBoard::diagonal(size);
    let mut next_left = true;

    let mut bot_l = ZeroBot::new(
        network(mapper, batch_size, path_l),
        settings_l,
        visits_l,
        StdRng::from_entropy(),
    );
    let mut bot_r = ZeroBot::new(
        network(mapper, batch_size, path_r),
        settings_r,
        visits_r,
        StdRng::from_entropy(),
    );

    let mut values_l = vec![];
    let mut values_r = vec![];

    while !board.is_done() {
        println!("{}", board);

        let tree_l = bot_l.build_tree(&board);
        let tree_r = bot_r.build_tree(&board);

        let sign = board.next_player().sign::<f32>(Player::A);
        values_l.push(sign * tree_l.values().wdl.value());
        values_r.push(sign * tree_r.values().wdl.value());

        println!("{}", tree_l.display(1, true, 100, false));
        println!("{}", tree_r.display(1, true, 100, false));

        let mv = if next_left {
            tree_l.best_move()
        } else {
            tree_r.best_move()
        };
        board.play(mv.unwrap()).unwrap();

        println!("{} played {:?}", if next_left { "left" } else { "right" }, mv);
        next_left ^= true;
    }

    println!("{}", board);
    println!("{:?}", board.outcome());

    println!("values_l={:?}", values_l);
    println!("values_r={:?}", values_r);
}

fn network<B: Board, M: BoardMapper<B>>(mapper: M, batch_size: usize, path: &str) -> CudaNetwork<B, M> {
    let graph = optimize_graph(
        &load_graph_from_onnx_path(path, false).unwrap(),
        OptimizerSettings::default(),
    );
    CudaNetwork::new(mapper, &graph, batch_size, Device::new(0))
}
