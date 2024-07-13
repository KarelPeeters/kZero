use std::time::Instant;

use board_game::games::chess::ChessBoard;
use board_game::util::board_gen::random_board_with_moves;
use itertools::Itertools;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use kn_cuda_sys::wrapper::handle::Device;
use kn_graph::onnx::load_graph_from_onnx_path;
use kn_graph::optimizer::optimize_graph;
use kz_core::mapping::chess::ChessStdMapper;
use kz_core::network::cudnn::CudaNetwork;
use kz_core::network::job_channel::job_pair;
use kz_core::network::multibatch::MultiBatchNetwork;
use kz_core::network::Network;
use kz_core::zero::node::UctWeights;
use kz_core::zero::step::{FpuMode, QMode};
use kz_core::zero::wrapper::{AsyncZeroBot, ZeroSettings};
use kz_misc::eval::tournament::{box_bot, run_tournament, BoxBotFn};
use kz_selfplay::server::executor::{batched_executor_loop, RunCondition};
use kz_util::math::ceil_div;

fn main() {
    let device = Device::new(0);

    let eval_batch_sizes = &[64, 256, 768, 1024];
    let search_batch_size = 16;
    let visits = 512;

    let all_settings = [
        (
            "loss-1",
            ZeroSettings::new(
                search_batch_size,
                UctWeights::default(),
                QMode::wdl(),
                FpuMode::Relative(0.0),
                FpuMode::Relative(0.0),
                1.0,
                1.0,
            ),
        ),
        (
            "loss-0",
            ZeroSettings::new(
                search_batch_size,
                UctWeights::default(),
                QMode::wdl(),
                FpuMode::Relative(0.0),
                FpuMode::Relative(0.0),
                0.0,
                1.0,
            ),
        ),
    ];

    let mapper = ChessStdMapper;

    let mut rng = StdRng::from_entropy();
    let positions = (0..100)
        .map(|_| random_board_with_moves(&ChessBoard::default(), rng.gen_range(0..4), &mut rng))
        .collect_vec();

    let mut bots: Vec<(_, BoxBotFn<ChessBoard>)> = vec![];

    let (fill_sender, fill_receiver) = flume::unbounded::<(usize, usize)>();
    let max_eval_batch_size = eval_batch_sizes.iter().copied().max().unwrap();

    let path = r#"C:\Documents\Programming\STTT\kZero\data\networks\chess_16x128_gen3634.onnx"#;

    let graph = optimize_graph(&load_graph_from_onnx_path(path, false).unwrap(), Default::default());

    let (client, server) = job_pair(4 * ceil_div(max_eval_batch_size, search_batch_size));
    let fill_sender = fill_sender.clone();

    std::thread::Builder::new()
        .name(format!("executor"))
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

    for (name, settings) in all_settings {
        let client = client.clone();
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

    println!("{}", result);
}
