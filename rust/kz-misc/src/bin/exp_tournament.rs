use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Instant;

use board_game::games::ataxx::AtaxxBoard;
use board_game::util::board_gen::random_board_with_moves;
use itertools::Itertools;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use kn_cuda_sys::wrapper::handle::Device;
use kn_graph::onnx::load_graph_from_onnx_path;
use kn_graph::optimizer::optimize_graph;
use kz_core::mapping::ataxx::AtaxxStdMapper;
use kz_core::network::cudnn::CudaNetwork;
use kz_core::network::job_channel::job_pair;
use kz_core::network::multibatch::MultiBatchNetwork;
use kz_core::network::Network;
use kz_core::zero::node::UctWeights;
use kz_core::zero::step::{FpuMode, QMode};
use kz_core::zero::wrapper::{AsyncZeroBot, ZeroSettings};
use kz_misc::convert::pt_to_onnx::convert_pt_to_onnx;
use kz_misc::eval::tournament::{box_bot, run_tournament, BoxBotFn};
use kz_selfplay::server::executor::{batched_executor_loop, RunCondition};
use kz_selfplay::server::start_pos::ataxx_start_pos;
use kz_util::math::ceil_div;

fn main() {
    let device = Device::new(0);

    let eval_batch_sizes = &[64, 256, 512, 512 + 256];
    let search_batch_size = 8;
    let visits = 512;

    let default_settings = ZeroSettings::simple(
        search_batch_size,
        UctWeights::default(),
        QMode::wdl(),
        FpuMode::Relative(0.0),
    );

    let mapper = AtaxxStdMapper::new(7);
    let game_str = format!("ataxx-{}", mapper.size());

    let target_pos_count = 40;

    // take most/least balanced book positions
    // let book_path = r#"C:\Documents\Programming\STTT\kZero\ignored\opening_book\ataxx-7-openings-8192.txt"#;
    // let mut book_positions = load_ataxx_book(book_path);
    // book_positions.sort_by_key(|&(score, _)| NotNan::from_inner((0.2 - score.abs()).abs()));
    //
    // let selected = book_positions
    //     .iter()
    //     .cloned()
    //     .unique_by(|(_, board)| board.clone())
    //     .take(target_pos_count)
    //     .collect_vec();
    // println!("Selected boards");
    // for (score, pos) in &selected {
    //     println!("  {}: {:?}", score, pos);
    // }
    // let positions = selected.iter().map(|(_, board)| board.clone()).collect_vec();

    let mut rng = StdRng::from_entropy();
    let start_pos = ataxx_start_pos(mapper.size(), "random-gaps-v1");
    let mut positions = HashSet::new();
    while positions.len() < target_pos_count {
        let pos = start_pos(&mut rng);

        // if pos.gaps().any() {
        //     println!("{}", pos);
        //     positions.insert(pos);
        // }

        let pos = if pos == AtaxxBoard::diagonal(mapper.size()) {
            random_board_with_moves(&pos, rng.gen_range(2..4), &mut rng)
        } else {
            pos
        };

        // let start = AtaxxBoard::diagonal(mapper.size());
        // let pos = random_board_with_moves(&start, rng.gen_range(2..4), &mut rng);

        positions.insert(pos);
    }
    let positions = positions.into_iter().collect_vec();

    println!(
        "Unique count: {}/{}",
        positions.iter().unique().count(),
        target_pos_count
    );
    println!(
        "Gap fraction: {}/{}",
        positions.iter().filter(|p| p.gaps().any()).count(),
        target_pos_count
    );

    let (fill_sender, fill_receiver) = flume::unbounded::<(usize, usize)>();
    let max_eval_batch_size = eval_batch_sizes.iter().copied().max().unwrap();

    let mut bots: Vec<(_, BoxBotFn<AtaxxBoard>)> = vec![
        // (
        // "mm-2".to_owned(),
        // Box::new(|| {
        //     Box::new(WrapAsync(MiniMaxBot::new(
        //         2,
        //         AtaxxTileHeuristic::default(),
        //         SmallRng::from_entropy(),
        //     )))
        // }),
    // )
    ];

    let paths = vec![
        (
            "old-1364",
            r#"C:\Documents\Programming\STTT\kZero\data\networks\ataxx_16x128_gen_1364.pt"#,
            default_settings,
        ),
        // (
        //     "old-v2-6897",
        //     r#"\\192.168.0.10\Documents\Karel A0\loop\ataxx-7\16-128-v2\training\gen_6897\network.pt"#,
        //     default_settings,
        // ),
        (
            "gaps-6972",
            r#"\\192.168.0.10\Documents\Karel A0\loop\ataxx-7\16x128_gaps\training\gen_6972\network.pt"#,
            default_settings,
        ),
    ];

    for (name, path, settings) in paths {
        if path.ends_with(".pt") {
            convert_pt_to_onnx(path, &game_str);
        }

        let path = PathBuf::from(path).with_extension("onnx");
        let graph = optimize_graph(&load_graph_from_onnx_path(path, false).unwrap(), Default::default());

        let (client, server) = job_pair(4 * ceil_div(max_eval_batch_size, search_batch_size));
        let fill_sender = fill_sender.clone();

        std::thread::Builder::new()
            .name(format!("executor-{}", name))
            .spawn(move || {
                let (graph_sender, graph_receiver) = flume::bounded(1);
                graph_sender.send(Some(graph)).unwrap();
                drop(graph_sender);

                batched_executor_loop(
                    max_eval_batch_size,
                    RunCondition::Any,
                    graph_receiver,
                    server,
                    |graph| {
                        MultiBatchNetwork::build_sizes(eval_batch_sizes, |size| {
                            CudaNetwork::new(mapper, &graph, size, device)
                        })
                    },
                    |network, batch_x| {
                        let result = network.evaluate_batch(&batch_x);
                        let max_size = network.used_batch_size(batch_x.len());
                        fill_sender.send((batch_x.len(), max_size)).unwrap();
                        result
                    },
                );
            })
            .unwrap();

        bots.push((
            format!("zero-{}", name),
            box_bot(move || AsyncZeroBot::new(client.clone(), settings, visits, StdRng::from_entropy())),
        ));
    }

    let on_print = {
        let mut prev = Instant::now();

        let mut total_filled = 0;

        move || {
            let mut delta_filled = 0;
            let mut delta_potential = 0;

            for (filled, potential) in fill_receiver.try_iter() {
                total_filled += filled;
                delta_filled += filled;
                delta_potential += potential;
            }

            let now = Instant::now();
            let delta = (now - prev).as_secs_f32();
            prev = now;

            let throughput = delta_potential as f32 / delta;
            let fill = delta_filled as f32 / delta_potential as f32;

            println!(
                "  throughput: {} evals/s, fill {} => {} evals",
                throughput, fill, total_filled
            );
        }
    };

    let result = run_tournament(bots, positions, Some(6), false, true, on_print);

    println!("Rounds:");
    for round in &result.rounds {
        println!("  Round {:?}:", round.id);
        println!("    start: {:?}", round.start);
        println!("    moves: {:?}", round.moves);
        println!("    outcome: {:?}", round.outcome);
    }

    println!("Result:");
    println!("{}", result);
}

#[allow(dead_code)]
fn load_ataxx_book(book_path: &str) -> Vec<(f32, AtaxxBoard)> {
    std::fs::read_to_string(book_path)
        .unwrap()
        .lines()
        .skip(1)
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let parts = line.split(',').map(str::trim).collect_vec();
            assert_eq!(parts.len(), 5);
            let value: f32 = parts[2].parse().unwrap();
            let fen = parts[3];
            let fen = &fen[1..fen.len() - 1];
            (value, AtaxxBoard::from_fen(fen).unwrap())
        })
        .collect_vec()
}
