use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::time::Instant;

use board_game::games::ataxx::AtaxxBoard;
use rand::rngs::StdRng;
use rand::SeedableRng;

use kn_cuda_sys::wrapper::handle::Device;
use kn_graph::onnx::load_graph_from_onnx_path;
use kn_graph::optimizer::optimize_graph;
use kz_core::mapping::ataxx::AtaxxStdMapper;
use kz_core::network::cudnn::CudaNetwork;
use kz_core::network::job_channel::job_pair;
use kz_core::network::Network;
use kz_core::zero::node::UctWeights;
use kz_core::zero::step::{FpuMode, QMode};
use kz_core::zero::wrapper::{AsyncZeroBot, ZeroSettings};
use kz_misc::eval::tournament::{box_bot, run_tournament, BoxBotFn};
use kz_selfplay::server::executor::{batched_executor_loop, RunCondition};
use kz_util::math::ceil_div;

fn main() -> std::io::Result<()> {
    let positions_path = r#"C:\Documents\Programming\STTT\kZero\ignored\opening_book\selected.txt"#;
    let network_path =
        r#"C:\Documents\Programming\STTT\kZero\data\loop\ataxx-7\16-128-v2\tmp\curr_network\network_5012.onnx"#;
    let output_path = r#"selected_evals.txt"#;

    let device = Device::new(0);

    let eval_batch_size = 1024;
    let search_batch_size = 16;
    let visits = 2048;
    let settings = ZeroSettings::simple(
        search_batch_size,
        UctWeights::default(),
        QMode::wdl(),
        FpuMode::Relative(0.0),
    );

    let mapper = AtaxxStdMapper::new(7);
    let graph = optimize_graph(
        &load_graph_from_onnx_path(network_path, false).unwrap(),
        Default::default(),
    );

    let mut positions = vec![];
    for line in std::fs::read_to_string(positions_path)?.lines() {
        let line = line.trim();
        if line.len() == 0 {
            continue;
        }
        positions.push(AtaxxBoard::from_fen(line).unwrap());
    }

    let (fill_sender, fill_receiver) = flume::unbounded();
    let (client, server) = job_pair(2 * ceil_div(eval_batch_size, search_batch_size));

    let mut bots: Vec<(_, BoxBotFn<AtaxxBoard>)> = vec![];
    for _ in 0..2 {
        let client = client.clone();
        bots.push((
            "zero-0",
            box_bot(move || AsyncZeroBot::new(client.clone(), settings, visits, StdRng::from_entropy())),
        ))
    }

    drop(client);

    std::thread::Builder::new()
        .name("executor".into())
        .spawn(move || {
            let (graph_sender, graph_receiver) = flume::bounded(1);
            graph_sender.send(Some(graph)).unwrap();
            drop(graph_sender);

            let max_batch_size = eval_batch_size;
            batched_executor_loop(
                max_batch_size,
                RunCondition::Any,
                graph_receiver,
                server,
                |graph| CudaNetwork::new(mapper, &graph, max_batch_size, device),
                move |network, batch_x| {
                    let result = network.evaluate_batch(&batch_x);
                    fill_sender.send((batch_x.len(), max_batch_size)).unwrap();
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
