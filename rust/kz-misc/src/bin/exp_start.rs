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

    let batch_size = 1;
    let mut network = CudaNetwork::new(mapper, &graph, batch_size, Device::new(0));

    let board = ChessBoard::default();
}
