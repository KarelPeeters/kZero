#![allow(unused_imports)]

use std::fs::read_to_string;
use std::time::Instant;

use board_game::board::{Board, BoardMoves};
use board_game::games::ataxx::AtaxxBoard;
use board_game::games::chess::{chess_game_to_pgn, ChessBoard, Rules};
use board_game::util::board_gen::{random_board_with_forced_win, random_board_with_moves};
use board_game::util::bot_game;
use board_game::util::game_stats::average_game_stats;
use internal_iterator::InternalIterator;
use itertools::Itertools;
use rand::rngs::StdRng;
use rand::{thread_rng, SeedableRng};

use kn_cuda_eval::Device;
use kn_graph::onnx::load_graph_from_onnx_path;
use kn_graph::optimizer::optimize_graph;
use kz_core::mapping::ataxx::AtaxxStdMapper;
use kz_core::mapping::chess::ChessStdMapper;
use kz_core::network::cudnn::CudaNetwork;
use kz_core::zero::node::UctWeights;
use kz_core::zero::step::{FpuMode, QMode};
use kz_core::zero::wrapper::{ZeroBot, ZeroSettings};
use kz_util::throughput::PrintThroughput;

fn main() {
    let weights = UctWeights::default();
    let settings = ZeroSettings::simple(1, weights, QMode::wdl(), FpuMode::Relative(0.0));

    let mapper = ChessStdMapper;
    let path = read_to_string("ignored/network_path.txt").unwrap();
    let graph = optimize_graph(&load_graph_from_onnx_path(path, false).unwrap(), Default::default());
    let mut network = CudaNetwork::new(mapper, &graph, settings.batch_size, Device::new(0));

    let mut rng = StdRng::from_entropy();

    let board = ChessBoard::default();
    let visits = 1;

    let tree = settings.build_tree(&board, &mut network, &mut rng, |tree| tree.root_visits() >= visits);

    println!("{}", tree.display(usize::MAX, true, usize::MAX, true));
}
