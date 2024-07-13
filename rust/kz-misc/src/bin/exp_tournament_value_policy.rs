use board_game::ai::simple::RandomBot;
use std::collections::HashSet;
use std::time::Instant;

use board_game::board::{Board, Player};
use board_game::games::ataxx::AtaxxBoard;
use board_game::games::chess::{chess_game_to_pgn, ChessBoard};
use board_game::util::board_gen::random_board_with_moves;
use itertools::Itertools;
use kn_cuda_sys::wrapper::handle::Device;
use kn_graph::onnx::load_graph_from_onnx_path;
use kn_graph::optimizer::optimize_graph;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use kz_core::bot::{MaxPolicyBot, MaxValueBot, WrapAsync};
use kz_core::mapping::ataxx::AtaxxStdMapper;
use kz_core::mapping::chess::ChessStdMapper;
use kz_core::network::cudnn::CudaNetwork;
use kz_core::network::job_channel::job_pair;
use kz_core::network::multibatch::MultiBatchNetwork;
use kz_core::network::Network;
use kz_core::zero::step::QMode;
use kz_misc::convert::pt_to_onnx::convert_pt_to_onnx;
use kz_misc::eval::tournament::{box_bot, run_tournament, BoxBotFn};
use kz_selfplay::server::executor::{batched_executor_loop, RunCondition};

fn main() {
    let device = Device::new(0);
    // let mapper = ChessStdMapper;
    let mapper = AtaxxStdMapper::new(7);

    let batch_sizes = vec![1024, 128, 8];
    let target_pos_count = 200;

    let mut rng = StdRng::from_entropy();
    let mut positions = HashSet::new();
    while positions.len() < target_pos_count {
        // let pos = ChessBoard::default();
        let pos = AtaxxBoard::diagonal(mapper.size());
        let pos = random_board_with_moves(&pos, rng.gen_range(2..4), &mut rng);

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

    let (fill_sender, fill_receiver) = flume::unbounded::<(usize, usize)>();

    let paths = vec![
        // (
        //     "8x64",
        //     r"\\192.168.0.10\Documents\Karel A0\loop\ataxx-7\8x64\training\gen_529\network.onnx",
        // ),
        (
            "16x128",
            r"C:\Documents\Programming\STTT\kZero\data\networks\ataxx_16x128_gen_1364.onnx",
        ),
        (
            "16x128-v2",
            r"\\192.168.0.10\Documents\Karel A0\loop\ataxx-7\16-128-v2\tmp\curr_network\network_6897.onnx",
        ),
        (
            "16x128_gaps",
            r"\\192.168.0.10\Documents\Karel A0\loop\ataxx-7\16x128_gaps\training\gen_6972\network.onnx",
        ),
    ];

    let mut bots: Vec<(String, BoxBotFn<AtaxxBoard>)> = vec![(
        "random".to_owned(),
        box_bot(|| WrapAsync(RandomBot::new(StdRng::from_entropy()))),
    )];
    let mut executors = vec![];

    for (name, path) in paths {
        if path.ends_with(".pt") {
            // convert_pt_to_onnx(path, "chess");
            convert_pt_to_onnx(path, "ataxx-7");
        }
        let graph = optimize_graph(&load_graph_from_onnx_path(path, false).unwrap(), Default::default());

        // start executor
        let batch_sizes = batch_sizes.clone();
        let max_batch_size = batch_sizes.iter().copied().max().unwrap_or(0);
        let (eval_client, eval_server) = job_pair(max_batch_size * 2);
        let fill_sender = fill_sender.clone();

        let executor = std::thread::Builder::new()
            .name(format!("executor-{}", name))
            .spawn(move || {
                let (graph_sender, graph_receiver) = flume::bounded(1);
                graph_sender.send(Some(graph)).unwrap();
                drop(graph_sender);

                batched_executor_loop(
                    max_batch_size,
                    RunCondition::Any,
                    graph_receiver,
                    eval_server,
                    |graph| {
                        MultiBatchNetwork::build_sizes(&batch_sizes, |size| {
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
        executors.push(executor);

        let eval_client_1 = eval_client.clone();
        let eval_client_2 = eval_client.clone();

        // bots.push((
        //     format!("{}-argmax(value)", name),
        //     box_bot(move || MaxValueBot {
        //         eval_client: eval_client_1.clone(),
        //         rng: StdRng::from_entropy(),
        //         q_mode: QMode::Value,
        //     }),
        // ));
        bots.push((
            format!("{}-argmax(wdl)", name),
            box_bot(move || MaxValueBot {
                eval_client: eval_client_2.clone(),
                rng: StdRng::from_entropy(),
                q_mode: QMode::wdl(),
            }),
        ));
        bots.push((
            format!("{}-argmax(policy)", name),
            box_bot(move || MaxPolicyBot {
                eval_client: eval_client.clone(),
                rng: StdRng::from_entropy(),
            }),
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
        // println!("  Round {:?}:", round.id);
        // println!("    start: {:?}", round.start);
        // println!("    moves: {:?}", round.moves);
        // println!("    outcome: {:?}", round.outcome);

        // chess_game_to_pgn("white","black")

        let (white_id, black_id) = match round.start.next_player() {
            Player::A => (round.id.i, round.id.j),
            Player::B => (round.id.j, round.id.i),
        };
        let name_white = &result.bot_names[white_id];
        let name_black = &result.bot_names[black_id];

        // println!("[Event \"{:?}\"]", round.id);
        // println!(
        //     "{}",
        //     chess_game_to_pgn(name_white, name_black, &round.start, &round.moves)
        // );
    }

    println!("Result:");
    println!("{}", result);

    println!("Waiting for executor to finish...");
    for e in executors {
        e.join().unwrap();
    }
}
