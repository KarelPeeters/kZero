#![allow(dead_code)]

use std::collections::HashSet;
use std::hash::Hash;

use board_game::board::Board;
use board_game::games::ataxx::AtaxxBoard;
use board_game::util::board_gen::random_board_with_moves;
use rand::rngs::SmallRng;
use rand::SeedableRng;

use kn_cuda_sys::wrapper::handle::Device;
use kn_graph::onnx::load_graph_from_onnx_path;
use kn_graph::optimizer::optimize_graph;
use kz_core::mapping::ataxx::AtaxxStdMapper;
use kz_core::network::cudnn::CudaNetwork;
use kz_core::zero::node::UctWeights;
use kz_core::zero::step::{FpuMode, QMode};
use kz_core::zero::tree::Tree;
use kz_core::zero::wrapper::ZeroSettings;

fn main() {
    let mut rng = SmallRng::seed_from_u64(2);

    // let cond = |board: &AtaxxBoard| {
    //     board.next_player() == Player::A
    //         && !board.is_done()
    //         && board.available_moves().count() == 2
    //         && board.clone_and_play(board.available_moves().next().unwrap()).outcome()
    //             == Some(Outcome::WonBy(Player::A))
    //         && board
    //             .clone_and_play(board.available_moves().skip(1).next().unwrap())
    //             .outcome()
    //             == None
    // };
    // let board = random_board_with_condition(&AtaxxBoard::diagonal(7), &mut rng, cond);

    let mapper = AtaxxStdMapper::new(7);
    let board = AtaxxBoard::diagonal(7);

    let board = random_board_with_moves(&board, 16, &mut rng);

    println!("{}", board);

    let batch_size = 64;

    let settings = ZeroSettings::new(
        batch_size,
        UctWeights::default(),
        QMode::wdl(),
        FpuMode::Relative(0.0),
        FpuMode::Relative(0.0),
        1.0,
        1.0,
    );

    // let mut network = DummyNetwork;

    let path = r"\\192.168.0.10\Documents\Karel A0\loop\ataxx-7\16x128_gaps\tmp\network_6972.onnx";
    let graph = optimize_graph(&load_graph_from_onnx_path(path, false).unwrap(), Default::default());
    let mut network = CudaNetwork::new(mapper, &graph, batch_size, Device::new(0));

    let visits = 100_000;
    let tree = settings.build_tree(&board, &mut network, &mut rng, |tree| tree.root_visits() >= visits);
    println!("{}", tree.display(1, true, usize::MAX, false));

    // find duplicate nodes
    let mut set_exact = HashSet::new();
    let mut set_sym = HashSet::new();
    count_nodes(&tree, &mut set_exact, &mut set_sym, 0, &board);

    println!("Tree total node count: {}", tree.len());
    println!(
        "Tree visited node count: {}",
        (0..tree.len()).filter(|&i| tree[i].complete_visits > 0).count()
    );
    println!("Tree visits: {}", tree.root_visits());

    println!("Distinct node count (exact): {}", set_exact.len());
    println!("Distinct node count (sym): {}", set_sym.len());
}

fn count_nodes<B: Board + Hash>(
    tree: &Tree<B>,
    set_exact: &mut HashSet<B>,
    set_sym: &mut HashSet<B>,
    node: usize,
    board: &B,
) {
    if tree[node].complete_visits == 0 || matches!(tree[node].outcome(), Ok(Some(_))) {
        return;
    }

    set_exact.insert(board.clone());
    set_sym.insert(board.canonicalize());

    for child in tree[node].children.unwrap() {
        let child_mv = tree[child].last_move.unwrap();
        let child_board = board.clone_and_play(child_mv).unwrap();
        count_nodes(tree, set_exact, set_sym, child, &child_board);
    }
}
