use std::borrow::Cow;

use board_game::board::{Board, BoardMoves};
use board_game::games::ataxx::AtaxxBoard;
use board_game::games::go::{GoBoard, Komi, Rules};
use board_game::util::board_gen::random_board_with_condition;
use board_game::wdl::OutcomeWDL;
use internal_iterator::InternalIterator;
use itertools::Itertools;
use rand::rngs::{SmallRng, StdRng};
use rand::{thread_rng, SeedableRng};

use kn_cuda_sys::wrapper::handle::Device;
use kn_graph::graph::Graph;
use kn_graph::onnx::load_graph_from_onnx_path;
use kn_graph::optimizer::optimize_graph;
use kz_core::mapping::ataxx::AtaxxStdMapper;
use kz_core::mapping::go::GoStdMapper;
use kz_core::mapping::PolicyMapper;
use kz_core::network::cudnn::CudaNetwork;
use kz_core::network::symmetry::AverageSymmetryNetwork;
use kz_core::network::{Network, ZeroEvaluation};
use kz_core::zero::node::UctWeights;
use kz_core::zero::step::{FpuMode, QMode};
use kz_core::zero::values::ZeroValuesPov;
use kz_core::zero::wrapper::ZeroSettings;
use kz_selfplay::binary_output::BinaryOutput;
use kz_selfplay::server::start_pos::go_start_pos;
use kz_selfplay::simulation::{Position, Simulation};

fn main() {
    let path = r#"\\192.168.0.10\Documents\Karel A0\loop\go-9\first\tmp\network_716.onnx"#;
    let size = 9;

    let graph = optimize_graph(&load_graph_from_onnx_path(path, false).unwrap(), Default::default());

    main_plot_komi(&graph, size);
    // main_play_game(&graph, size);

    //
    // let settings = ZeroSettings::simple(batch_size, UctWeights::default(), QMode::wdl(), FpuMode::Fixed(0.0));
    //
    // let graph = optimize_graph(&load_graph_from_onnx_path(path, false).unwrap(), Default::default());
    // let mut network = CudaNetwork::new(mapper, &graph, batch_size, Device::new(0));

    // let mut iter = RandomBoardIterator::new(GoBoard::new(size, 15, Rules::cgos()), thread_rng())
    //     .unwrap()
    //     .filter(|b| !b.is_done());
    //
    // for _ in 0..1024 {
    //     let batch_real = (&mut iter).take(batch_size).collect_vec();
    //
    //     let batch_tt = batch_real
    //         .iter()
    //         .map(|b| clear_board(b, Rules::tromp_taylor()))
    //         .collect_vec();
    //     let batch_cgos = batch_real.iter().map(|b| clear_board(b, Rules::cgos())).collect_vec();
    //
    //     let evals_tt = network.evaluate_batch(&batch_tt);
    //     let evals_cgos = network.evaluate_batch(&batch_cgos);
    //
    //     for i in 0..batch_size {
    //         let eval_tt = &evals_tt[i];
    //         let eval_cgos = &evals_cgos[i];
    //         let board = &batch_real[i];
    //
    //         let available_tt: Vec<_> = batch_tt[i].available_moves().unwrap().collect();
    //         let available_cgos: Vec<_> = batch_cgos[i].available_moves().unwrap().collect();
    //         if available_tt != available_cgos {
    //             continue;
    //         }
    //
    //         // if (eval_tt.values.value.value - eval_cgos.values.value.value).abs() > 0.2 {
    //         //     println!("{}", board);
    //         //     println!("  TT: {}", eval_tt.values);
    //         //     println!("  CGOS: {}", eval_cgos.values);
    //         // }
    //
    //         let kdl =
    //             kdl_divergence(&eval_tt.policy, &eval_cgos.policy) + kdl_divergence(&eval_cgos.policy, &eval_tt.policy);
    //         let max_diff = zip_eq_exact(eval_tt.policy.as_ref(), eval_cgos.policy.as_ref())
    //             .map(|(a, b)| N32::from_inner((a - b).abs()))
    //             .max()
    //             .unwrap()
    //             .into_inner();
    //
    //         // println!("div: {}", div);
    //         // board.available_moves().unwrap().enumerate().for_each(|(i, mv)| {
    //         //     println!("  {} {} {}", mv, eval_tt.policy[i], eval_cgos.policy[i]);
    //         // });
    //
    //         if max_diff > 0.2 {
    //             println!("{}", board);
    //
    //             println!("kdl: {}, diff: {}", kdl, max_diff);
    //
    //             board.available_moves().unwrap().enumerate().for_each(|(i, mv)| {
    //                 let diff = (eval_tt.policy[i] - eval_cgos.policy[i]).abs();
    //                 let extra = if diff == max_diff { "* " } else { "" };
    //                 println!("  {} {} {}{}", mv, eval_tt.policy[i], eval_cgos.policy[i], extra);
    //             });
    //         }
    //     }
    // }

    // let mut board = GoBoard::new(size, 0, Rules::tromp_taylor());
    // let mut rng = StdRng::seed_from_u64(0);
    // let mut network = AverageSymmetryNetwork::new(network);
    //
    // while !board.is_done() {
    //     println!("{}", board);
    //     let tree = settings.build_tree(&board, &mut network, &mut rng, |tree| tree.root_visits() >= visits);
    //     println!("{}", tree.display(1, false, usize::MAX, false));
    //     board.play(tree.best_move().unwrap()).unwrap();
    // }
}

fn main_play_game(graph: &Graph, size: u8) {
    let mapper = GoStdMapper::new(size, false);

    let visits = 2048;
    let batch_size = 16;
    let settings = ZeroSettings::simple(batch_size, UctWeights::default(), QMode::wdl(), FpuMode::Fixed(1.0));

    let mut network = CudaNetwork::new(mapper, graph, batch_size, Device::new(0));
    let mut rng = SmallRng::seed_from_u64(0);

    let mut board = GoBoard::new(size, Komi::new(15), Rules::cgos());

    while !board.is_done() {
        println!("{}", board);

        let tree = settings.build_tree(&board, &mut network, &mut rng, |tree| tree.root_visits() >= visits);
        println!("{}", tree.display(1, true, 8, false));

        let mv = tree.best_move().unwrap();
        println!("playing {}", mv);

        board.play(mv).unwrap();
    }

    println!("{}", board);
    println!("{:?}", board.outcome());
}

fn main_plot_komi(graph: &Graph, size: u8) {
    let mapper = GoStdMapper::new(size, false);
    let mut network = CudaNetwork::new(mapper, &graph, 1, Device::new(0));

    let mut values = vec![];
    let mut komis = vec![];

    for komi_2 in -100..100 {
        let mut komi = None;
        for (rules_name, rules) in [("TT", Rules::tromp_taylor()), ("CGOS", Rules::cgos())] {
            let board = GoBoard::new(size, Komi::new(komi_2), rules);
            let eval = network.evaluate(&board);
            println!("rules={:>6} komi={:>6}: {}", rules_name, board.komi(), eval.values,);
            values.push(eval.values.value.value);

            komi = Some(board.komi());
        }

        komis.push(komi.unwrap());
    }

    println!("komis: {:?}", komis);
    println!("values: {:?}", values);
}

fn clear_board(board: &GoBoard, rules: Rules) -> GoBoard {
    GoBoard::from_parts(
        rules,
        board.chains().clone(),
        board.next_player(),
        board.state(),
        Default::default(),
        board.komi(),
    )
}

fn main_replay() {
    let moves = [
        33, 6, 22, 54, 35, 41, 39, 63, 42, 31, 30, 50, 23, 52, 44, 69, 51, 53, 61, 59, 57, 60, 17, 47, 56, 78, 67, 48,
        49, 58, 38, 46, 40, 65, 55, 66, 37, 48, 76, 18, 71, 45, 12, 26, 25, 8, 16, 77, 68, 36, 7, 27, 43, 1, 20, 28,
        10, 73, 47, 64, 21, 4, 15, 72, 62, 81, 48, 79, 80, 70, 9, 75, 8, 32, 61, 62, 67, 61, 14, 71, 19, 80, 29, 0, 76,
        68, 76, 67, 11, 0, 3, 13, 24, 2, 5, 1, 46, 74, 76, 72, 62, 53, 64, 70, 36, 32, 77, 60, 26, 68, 2, 52, 31, 81,
        13, 61, 45, 75, 34, 59, 78, 65, 6, 50, 41, 79, 54, 71, 73, 69, 1, 63, 18, 80, 74, 66, 67, 65, 66, 58, 27, 62,
        0, 58, 65, 72, 81, 63, 68, 59, 62, 70, 32, 60, 4, 69, 52, 61, 50, 71, 80, 79, 53, 80, 0, 81, 70, 0, 0,
    ];

    let mut board = GoBoard::new(9, Komi::new(16), Rules::tromp_taylor());
    println!("{}", board);

    for mv in moves {
        let mv = GoStdMapper::new(board.size(), false).index_to_move(&board, mv).unwrap();
        println!("{}", mv);
        board.play(mv).unwrap();
        println!("{}", board);
    }
}

fn main_starts() {
    let mut rng = StdRng::seed_from_u64(0);
    let start = go_start_pos(9, "default");
    for _ in 0..100 {
        println!("{:?}", start(&mut rng));
    }
}

fn main_out() {
    let mut out = BinaryOutput::new("test", "ataxx-7", AtaxxStdMapper::new(7)).unwrap();

    let board = random_board_with_condition(&AtaxxBoard::default(), &mut thread_rng(), |b| {
        b.children()
            .map_or(false, |children| children.any(|(_, c)| c.must_pass()))
    });

    let mv = board
        .available_moves()
        .unwrap()
        .find(|&mv| board.clone_and_play(mv).unwrap().must_pass())
        .unwrap();
    let next = board.clone_and_play(mv).unwrap();

    let mv_count = board.available_moves().unwrap().count();

    let eval = ZeroEvaluation {
        values: ZeroValuesPov::from_outcome(OutcomeWDL::Draw, 0.0),
        policy: Cow::Owned(vec![1.0 / mv_count as f32; mv_count]),
    };
    out.append(&Simulation {
        positions: vec![Position {
            board: board.clone(),
            is_full_search: false,
            played_mv: mv,
            zero_visits: 100,
            zero_evaluation: eval.clone(),
            net_evaluation: eval,
        }],
        final_board: next,
    })
    .unwrap();

    out.finish().unwrap();
}
