use board_game::games::chess::ChessBoard;
use itertools::Itertools;
use rand::rngs::SmallRng;
use rand::SeedableRng;

use kn_cuda_sys::wrapper::handle::Device;
use kn_graph::onnx::load_graph_from_onnx_path;
use kn_graph::optimizer::{optimize_graph, OptimizerSettings};
use kz_core::mapping::chess::ChessStdMapper;
use kz_core::network::cudnn::CudaNetwork;
use kz_core::zero::node::UctWeights;
use kz_core::zero::step::{FpuMode, QMode};
use kz_core::zero::wrapper::ZeroSettings;

fn main() {
    let path = r"C:\Documents\Programming\STTT\kZero\data\networks\chess_16x128_gen3634.onnx";
    let mapper = ChessStdMapper;
    let graph = optimize_graph(
        &load_graph_from_onnx_path(path, false).unwrap(),
        OptimizerSettings::default(),
    );

    let batch_size = 128;
    let mut network = CudaNetwork::new(mapper, &graph, batch_size, Device::new(0));
    let settings = ZeroSettings::simple(batch_size, UctWeights::default(), QMode::wdl(), FpuMode::Fixed(0.0));
    let visits = 100_000;
    let mut rng = SmallRng::seed_from_u64(0);

    let board = ChessBoard::default();

    let tree = settings.build_tree(&board, &mut network, &mut rng, |tree| tree.root_visits() >= visits);

    println!("{}", tree.display(1, true, usize::MAX, false));

    let mut visit_more = 0;
    let mut visit_once = 0;
    let mut visit_never = 0;

    for i in 0..tree.len() {
        let node = &tree[i];

        match node.complete_visits {
            0 => visit_never += 1,
            1 => visit_once += 1,
            _ => visit_more += 1,
        }
    }

    println!("visit_more: {}", visit_more as f32 / tree.len() as f32);
    println!("visit_once: {}", visit_once as f32 / tree.len() as f32);
    println!("visit_never: {}", visit_never as f32 / tree.len() as f32);
}
