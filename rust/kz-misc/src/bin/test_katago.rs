use board_game::board::{Board, BoardMoves, Player};
use board_game::games::ataxx::{AtaxxBoard, Move};
use cuda_nn_eval::executor::CudaExecutor;
use cuda_sys::wrapper::handle::Device;
use futures::FutureExt;
use internal_iterator::InternalIterator;
use itertools::Itertools;
use ndarray::{Array2, Array4, ArrayViewMut1, ArrayViewMut3, Axis};

use kz_core::network::common::softmax_in_place;
use nn_graph::cpu::cpu_eval_graph_exec;
use nn_graph::dot::graph_to_svg;
use nn_graph::graph::Graph;
use nn_graph::katago::load_katago_network_path;
use nn_graph::optimizer::optimize_graph;

fn main() {
    let path = r#"C:\Documents\Programming\STTT\kZero\data\katago-ataxx7x7.bin"#;
    let graph = load_katago_network_path(path, 7).unwrap();
    let graph = optimize_graph(&graph, Default::default());

    graph_to_svg("ignored/graphs/katago.svg", &graph, true).unwrap();

    // test_example(&graph);
    test_profile(&graph);
}

fn test_profile(graph: &Graph) {
    let batch_size = 1024;

    let device = Device::new(0);
    let mut exec = CudaExecutor::new(device, &graph, batch_size);
    exec.set_profile(true);

    let input_spatial = Array4::zeros((batch_size, 22, 7, 7)).into_shared().into_dyn();
    let input_global = Array2::zeros((batch_size, 19)).into_shared().into_dyn();
    let _ = exec.evaluate_tensors(&[input_spatial, input_global]);

    println!("{}", exec.last_profile().unwrap());
}

fn test_example(graph: &Graph) {
    // test graph and executors
    let batch_size = 1;

    let mut input_spatial = Array4::zeros((batch_size, 22, 7, 7));
    let mut input_global = Array2::zeros((batch_size, 19));

    let board = AtaxxBoard::diagonal(7);
    encode_ataxx_board(
        &board,
        input_spatial.index_axis_mut(Axis(0), 0),
        input_global.index_axis_mut(Axis(0), 0),
    );

    println!("{:?}", input_spatial);
    println!("{:?}", input_global);

    let inputs = [
        input_spatial.into_shared().into_dyn(),
        input_global.into_shared().into_dyn(),
    ];

    let outputs = cpu_eval_graph_exec(&graph, batch_size, &inputs, false).output_tensors();
    assert_eq!(outputs.len(), 4);

    let policy = &outputs[0];
    let wdl = &outputs[1];
    let misc = &outputs[2];
    let ownership = &outputs[3];

    let mut policy = policy.into_iter().copied().collect_vec();
    let mut wdl = wdl.into_iter().copied().collect_vec();

    println!("policy_logits: {:?}", policy);
    println!("wdl_logits: {:?}", wdl);

    for c in board.full_mask() {
        let i = 1 + c.dense_index(board.size()) as usize;

        if !board.is_done() {
            let available = match board.tile(c) {
                Some(p) => {
                    if p == board.next_player() {
                        board
                            .available_moves()
                            .unwrap()
                            .any(|mv| matches!(mv, Move::Jump { from, to: _ } if from == c))
                    } else {
                        false
                    }
                }
                None => board.is_available_move(Move::Copy { to: c }).unwrap(),
            };

            if !available {
                policy[i] = f32::NEG_INFINITY;
            }
        }
    }
    if !board.must_pass() {
        policy[0] = f32::NEG_INFINITY;
    }

    softmax_in_place(&mut policy);
    softmax_in_place(&mut wdl);

    for c in board.full_mask() {
        let i = 1 + c.dense_index(board.size()) as usize;
        println!("  move {} policy {}", c, policy[i]);
    }

    println!("policy = {:?}", policy);
    println!("wdl = {:?}", wdl);
    println!("misc = {:#?}", misc);
    println!("ownership = {:#?}", ownership);
}

fn encode_ataxx_board(board: &AtaxxBoard, mut spatial: ArrayViewMut3<f32>, mut global: ArrayViewMut1<f32>) {
    spatial.fill(0.0);
    global.fill(0.0);

    // TODO set selected
    let selected = None;

    for c in board.full_mask() {
        let values = &[
            1.0,
            (board.tile(c) == Some(board.next_player())) as u8 as f32,
            (board.tile(c) == Some(board.next_player().other())) as u8 as f32,
            board.gaps().has(c) as u8 as f32,
            (selected == Some(c)) as u8 as f32,
        ];

        for (channel, &v) in values.into_iter().enumerate() {
            spatial[(channel, c.y() as usize, c.x() as usize)] = v;
        }
    }

    // stage and must_pass
    if selected.is_none() {
        // stage == 0, single or double start
        global[1] = board.must_pass() as u8 as f32;
    } else {
        // stage == 1, double end
        global[0] = 1.0;
    }

    // rule LOOPDRAW_PASSCONTINUE
    global[2] = 1.0;

    // komi
    let komi: f32 = 0.0;
    let board_area = (board.size() as f32).powi(2);
    let self_komi = if board.next_player() == Player::A { komi } else { -komi };
    global[5] = (self_komi).tanh();
    global[6] = (self_komi * 0.3).tanh();
    global[7] = (self_komi * 0.1).tanh();
    global[8] = self_komi / board_area;
    global[9] = (komi + board_area) % 2.0;

    // playout doubling
    let playout_doubling_advantage = 0.0;
    global[15] = 1.0;
    global[16] = 0.5 * playout_doubling_advantage;

    let no_result_utility_for_white = 0.0;
    let no_result_utility_self = if board.next_player() == Player::B {
        no_result_utility_for_white
    } else {
        -no_result_utility_for_white
    };
    global[17] = no_result_utility_self;
}
