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
use kz_core::zero::step::{FpuMode, QMode};
use kz_core::zero::wrapper::ZeroSettings;

// const NETWORK: &[u8] =
//     include_bytes!(r#"C:\Documents\Programming\STTT\kZero\data\networks\ataxx_16x128_gen_1364.onnx"#);

const NETWORK: &[u8] =
    include_bytes!(r#"C:\Documents\Programming\STTT\kZero\data\networks\ataxx_16x128_gaps_gen_6972.onnx"#);

#[derive(clap::Parser)]
struct Args {
    #[clap(short, long)]
    network: PathBuf,
    #[clap(short, long)]
    batch_size: usize,
    #[clap(long)]
    max_nodes: Option<usize>,
}

// TODO tree reuse
// TODO write a proper script for this
fn main() {
    // let path = "C:/Documents/Programming/STTT/kZero/data/loop/ataxx-7/16x128/training/gen_1364/network.onnx";
    let graph = optimize_graph(&load_graph_from_onnx_bytes(NETWORK).unwrap(), Default::default());

    let weights = UctWeights {
        exploration_weight: 4.0,
        ..UctWeights::default()
    };
    let settings = ZeroSettings::simple(128, weights, QMode::wdl(), FpuMode::Relative(0.0));

    let mapper = AtaxxStdMapper::new(7);

    loop {
        let result = catch_unwind(|| {
            let mut network = CudaNetwork::new(mapper, &graph, settings.batch_size, Device::new(0));
            let mut rng = StdRng::from_entropy();

            // let stream = TcpStream::connect("server.ataxx.org:28028").unwrap();
            let stream = TcpStream::connect("ataxx.vanheusden.com:28028").unwrap();
            stream.set_read_timeout(Some(Duration::from_secs_f32(30.0))).unwrap();

            let mut writer = BufWriter::new(&stream);
            writeln!(&mut writer, "user kzero-test-v2").unwrap();
            writeln!(&mut writer, "pass 635A26AD425331A6").unwrap();
            writer.flush().unwrap();

            let e = uai::client::run(
                |board, time_to_use| {
                    let start = Instant::now();

                    let tree = settings.build_tree(board, &mut network, &mut rng, |_| {
                        start.elapsed().as_secs_f32() >= time_to_use
                    });

                    let mv = tree.best_move().unwrap();

                    let info = format!(
                        "nodes: {}, values: {:?}, depth {:?}",
                        tree.root_visits(),
                        tree.values(),
                        tree.depth_range(0)
                    );

                    (mv, info)
                },
                "kZero-test-v2",
                "KarelPeeters",
                &stream,
                &stream,
                &mut std::io::stdout().lock(),
                // "kZero",
                // "KarelPeeters",
                // &mut std::io::stdin().lock(),
                // &mut std::io::stdout().lock(),
                // File::create("log.txt").unwrap(),
            );

            if e.as_ref().err().map_or(false, |e| e.kind() == ErrorKind::TimedOut) {
                println!("Timeout");
            } else {
                e.unwrap();
            }
        });

        println!("{:?}", result);
        println!("recreating connection");
    }
}
