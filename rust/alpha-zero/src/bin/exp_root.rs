use std::str::FromStr;
use board_game::games::chess::{ChessBoard, Rules};
use board_game::util::board_gen::random_board_with_forced_win;
use chess::Board;
use rand::thread_rng;
use alpha_zero::mapping::chess::ChessStdMapper;

use alpha_zero::network::cudnn::CudnnNetwork;
use alpha_zero::zero::tree::Tree;
use alpha_zero::zero::wrapper::ZeroSettings;
use cuda_nn_eval::Device;
use nn_graph::onnx::load_graph_from_onnx_path;

fn main() {
    let path = "../data/new_loop_real/first/chess/initial_network.onnx";

    let batch_size = 256;
    let settings = ZeroSettings::new(batch_size, 2.0);
    let board_inner = Board::from_str("5rk1/1b3ppp/8/2RN4/8/8/2Q2PPP/6K1 w - - 0 1").unwrap();
    let board = ChessBoard::new(board_inner, Rules::default());
    println!("{}", board);

    let graph = load_graph_from_onnx_path(path);
    let mapper = ChessStdMapper;

    let mut network = CudnnNetwork::new(mapper, graph, batch_size, Device::new(0));

    let mut tree = Tree::new(board.clone());

    for i in 0..1000 {
        settings.expand_tree(&mut tree, &mut network, &(batch_size as u64 * (i + 1)));
        println!("{}", tree.display(1));
    }
}
