use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
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
use kz_misc::eval::tournament::{box_bot, run_tournament, BoxBotFn};
use kz_selfplay::server::executor::{batched_executor_loop, RunCondition};
use kz_util::math::ceil_div;

fn main() -> std::io::Result<()> {
    let network_path =
        r#"C:\Documents\Programming\STTT\kZero\data\loop\ataxx-7\16-128-v2\tmp\curr_network\network_5012.onnx"#;
    let output_path = r#"selected_evals.txt"#;

    let device = Device::new(0);

    let eval_batch_sizes = &[64, 1024];
    let search_batch_size = 16;
    let visits = 512;
    let settings = move |t| {
        ZeroSettings::new(
            search_batch_size,
            UctWeights::default(),
            QMode::wdl(),
            FpuMode::Relative(0.0),
            FpuMode::Relative(0.0),
            1.0,
            t,
        )
    };

    let mapper = AtaxxStdMapper::new(7);
    let graph = optimize_graph(
        &load_graph_from_onnx_path(network_path, false).unwrap(),
        Default::default(),
    );

    let mut rng = StdRng::from_entropy();
    let positions = (0..50)
        .map(|_| random_board_with_moves(&AtaxxBoard::default(), rng.gen_range(0..4), &mut rng))
        .collect_vec();

    let (fill_sender, fill_receiver) = flume::unbounded();
    let max_eval_batch_size = eval_batch_sizes.iter().copied().max().unwrap();
    let (client, server) = job_pair(2 * ceil_div(max_eval_batch_size, search_batch_size));

    let mut bots: Vec<(_, BoxBotFn<AtaxxBoard>)> = vec![];
    for t in [1.0, 1.5, 2.0] {
        let client = client.clone();
        bots.push((
            format!("zero-{}", t),
            box_bot(move || AsyncZeroBot::new(client.clone(), settings(t), visits, StdRng::from_entropy())),
        ))
    }
    drop(client);

    std::thread::Builder::new()
        .name("executor".into())
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
                    MultiBatchNetwork::build_sizes(eval_batch_sizes, |batch_size| {
                        CudaNetwork::new(mapper, &graph, batch_size, device)
                    })
                },
                move |network, batch_x| {
                    let result = network.evaluate_batch(&batch_x);
                    let max_size = network.used_batch_size(batch_x.len());
                    fill_sender.send((batch_x.len(), max_size)).unwrap();
                    result
                },
            );
        })
        .unwrap();

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

    let result = run_tournament(bots, positions, Some(4), false, false, on_print);
    println!("{}", result);

    let mut output = BufWriter::new(File::create(output_path)?);
    writeln!(&mut output, "len, outcome")?;
    for round in result.rounds {
        writeln!(
            &mut output,
            "{}, {}, {}",
            round.id.s,
            round.moves.len(),
            round.outcome.sign::<i8>().value_a
        )?;
    }

    Ok(())
}
