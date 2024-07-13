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

// average length before:                     102
// average length after change (wrong sign?): 144
// average length after change (right sign?): 112 -> black tries hard to stop white from winning?

//TODO why do things suddenly take so much time? is uct calculation that much slower now?

fn main() {
    let path = read_to_string("ignored/network_path.txt").unwrap();

    let board = ChessBoard::new_without_history_fen("8/p7/kpP5/qrp1b3/rpP2b2/pP4b1/P3K3/8 w - - 0 1", Rules::default());

    println!("===================================");
    println!("===================================");
    println!("{}", board);

    // for path in paths {
    println!("===================================");
    println!("Using network {}", path);

    let graph = optimize_graph(&load_graph_from_onnx_path(path, false).unwrap(), Default::default());

    let weights = UctWeights::default();
    let settings = ZeroSettings::simple(128, weights, QMode::wdl(), FpuMode::Relative(0.0));
    let visits = 100_000;

    let mapper = ChessStdMapper;
    let mut network = CudaNetwork::new(mapper, &graph, settings.batch_size, Device::new(0));
    let mut rng = StdRng::from_entropy();

    let start = Instant::now();

    println!("-------- Building tree");
    let tree = settings.build_tree(&board, &mut network, &mut rng, |tree| tree.root_visits() >= visits);

    println!("Building tree took {:.4}s", (Instant::now() - start).as_secs_f32());

    println!("-------- Outer display initial tree");

    println!("{}", tree.display(3, true, 100, true));

    println!("-------- Keep moves");
    let mut sub_tree = tree.keep_moves(&[tree.best_move().unwrap()]).unwrap();

    println!("-------- Outer display kept");
    println!("{}", sub_tree.display(3, true, 100, true));

    println!("-------- Expanding after keep");
    settings.expand_tree(&mut sub_tree, &mut network, &mut rng, |tree| {
        tree.root_visits() >= visits
    });

    println!("-------- Outer display expanded");
    println!("{}", sub_tree.display(3, true, 100, true));
}
