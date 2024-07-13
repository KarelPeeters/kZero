use board_game::board::Board;
use std::cmp::max;
use std::io::Write;
use std::io::{BufWriter, ErrorKind};
use std::net::TcpStream;
use std::panic::catch_unwind;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use board_game::interface::uai;
use rand::rngs::StdRng;
use rand::SeedableRng;

use kn_cuda_eval::Device;
use kn_graph::onnx::load_graph_from_onnx_bytes;
use kn_graph::optimizer::optimize_graph;
use kz_core::mapping::ataxx::AtaxxStdMapper;
use kz_core::network::cudnn::CudaNetwork;
use kz_core::zero::node::UctWeights;
use kz_core::zero::step::{zero_step_apply, FpuMode, QMode};
use kz_core::zero::wrapper::ZeroSettings;

const NETWORK: &[u8] =
    include_bytes!(r#"C:\Documents\Programming\STTT\kZero\data\networks\ataxx_16x128_gaps_gen_6972.onnx"#);

// TODO tree reuse
// TODO write a proper script for this
fn main() {
    let graph = optimize_graph(&load_graph_from_onnx_bytes(NETWORK).unwrap(), Default::default());

    let weights = UctWeights {
        exploration_weight: 2.0,
        ..UctWeights::default()
    };
    let settings = ZeroSettings::new(
        16,
        weights,
        QMode::wdl(),
        FpuMode::Fixed(1.0),
        FpuMode::Fixed(0.0),
        1.0,
        1.0,
    );

    let mapper = AtaxxStdMapper::new(7);

    // let mut network = CudaNetwork::new(mapper, &graph, settings.batch_size, Device::new(0));
    // let mut network = CPUN
    let mut rng = StdRng::from_entropy();

    uai::client::run(
        |board, time_to_use| {
            let start = Instant::now();
            let mut step_start = start;

            let tree = settings.build_tree(board, &mut network, &mut rng, |_| {
                let step_end = Instant::now();

                let step_elapsed = f32::max(0.001, (step_end - step_start).as_secs_f32());
                step_start = step_end;

                // add some margin for step variability
                let elapsed_after_next_step = start.elapsed().as_secs_f32() + step_elapsed * 2.0;
                elapsed_after_next_step >= time_to_use
            });

            let mv = tree
                .best_move()
                .unwrap_or_else(|| board.random_available_move(&mut rng).unwrap());

            let info = format!(
                "nodes: {}, values: {:?}, depth {:?}",
                tree.root_visits(),
                tree.values(),
                tree.depth_range(0)
            );

            (mv, info)
        },
        "kZero_gaps_6972",
        "KarelPeeters",
        &mut std::io::stdin().lock(),
        &mut std::io::stdout().lock(),
        std::fs::File::create("log.txt").unwrap(),
    )
    .unwrap();
}
