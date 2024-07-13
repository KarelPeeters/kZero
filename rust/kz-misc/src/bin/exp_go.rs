use board_game::board::{Board, BoardSymmetry};
use board_game::games::go::{GoBoard, Komi, Rules};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::collections::HashSet;

use kn_cuda_sys::wrapper::handle::Device;
use kn_graph::onnx::load_graph_from_onnx_path;
use kn_graph::optimizer::optimize_graph;
use kz_core::mapping::go::GoStdMapper;
use kz_core::network::cudnn::CudaNetwork;
use kz_core::network::Network;
use kz_core::zero::node::Node;
use kz_core::zero::step::{FpuMode, QMode};
use kz_core::zero::tree::Tree;
use kz_core::zero::wrapper::ZeroSettings;

// for komi_2 in -30..30 {
//     let komi = Komi::new(komi_2);
//
//     let board = GoBoard::new(9, komi, Rules::tromp_taylor());
//     let eval = network.evaluate(&board).values;
//     println!("{}: {}", komi, eval);
// }

fn main() {
    let path = r"C:\Documents\Programming\STTT\kZero\data\networks\go9_first_gen716.onnx";
    let mapper = GoStdMapper::new(9, false);

    let batch_size = 128;

    let device = Device::new(0);
    let graph = optimize_graph(&load_graph_from_onnx_path(path, false).unwrap(), Default::default());
    let mut network = CudaNetwork::new(mapper, &graph, batch_size, device);

    let settings = ZeroSettings::simple(batch_size, Default::default(), QMode::wdl(), FpuMode::Fixed(0.0));
    let visits = 1_000_000;

    let board = GoBoard::new(9, Komi::try_from(7.5).unwrap(), Rules::tromp_taylor());
    println!("{}", board);

    let mut rng = SmallRng::seed_from_u64(0);
    let tree = settings.build_tree(&board, &mut network, &mut rng, |tree| tree.root_visits() >= visits);

    println!("{}", tree.display(1, true, usize::MAX, false));

    // duplicate boards without history
    let mut boards = HashSet::new();
    let mut boards_histless = HashSet::new();
    let mut boards_histless_sym = HashSet::new();
    let mut eval_nodes = 0;

    visit_nodes(&tree, |ni, b| {
        let node = &tree[ni];

        if node.complete_visits > 0 {
            boards.insert(b.clone());
            boards_histless.insert(b.clone_without_history());
            boards_histless_sym.insert(b.clone_without_history().canonicalize());
            eval_nodes += 1;
        }
    });

    println!("Eval_nodes: {}", eval_nodes);
    println!("Unique boards: {}", boards.len());
    println!("Unique boards without history: {}", boards_histless.len());
}

fn visit_nodes<B: Board>(tree: &Tree<B>, mut f: impl FnMut(usize, &B)) {
    fn rec<B: Board>(tree: &Tree<B>, f: &mut impl FnMut(usize, &B), index: usize, board: &B) {
        f(index, board);

        let node = &tree[index];
        if let Some(children) = node.children {
            for child_index in children {
                let mv = tree[child_index].last_move.unwrap();
                let child_board = board.clone_and_play(mv).unwrap();
                rec(tree, f, child_index, &child_board);
            }
        }
    }

    rec(tree, &mut f, 0, tree.root_board())
}
