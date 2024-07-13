use std::fs::File;
use std::io::{BufWriter, Write};

use board_game::games::ataxx::AtaxxBoard;
use board_game::util::game_stats::all_possible_boards;

use kn_cuda_sys::wrapper::handle::Device;
use kn_graph::onnx::load_graph_from_onnx_path;
use kn_graph::optimizer::{optimize_graph, OptimizerSettings};
use kz_core::mapping::ataxx::AtaxxStdMapper;
use kz_core::zero::node::UctWeights;
use kz_core::zero::step::{FpuMode, QMode};
use kz_core::zero::wrapper::ZeroSettings;
use kz_misc::eval::batch_tree_eval::{batch_tree_eval, BatchEvalSettings};

fn main() -> std::io::Result<()> {
    let path = r#"\\192.168.0.10\Documents\Karel A0\loop\ataxx-7\16-128-v2\tmp\curr_network\network_4460.onnx"#;
    // let path = r#"/workspace/kZero/data/loop/ataxx-7/16-128-v2/tmp/curr_network/network_4932.onnx"#;
    let graph = optimize_graph(
        &load_graph_from_onnx_path(path, false).unwrap(),
        OptimizerSettings::default(),
    );
    let mapper = AtaxxStdMapper::new(7);

    let zero_settings = ZeroSettings::simple(32, UctWeights::default(), QMode::wdl(), FpuMode::Relative(0.0));

    let settings = BatchEvalSettings {
        visits: 2048,
        network_batch_size: 2048,
        cpu_threads: 4,
        max_concurrent_positions: 1024,
    };

    let max_depth = 4;

    let mut positions = vec![];
    let mut depths = vec![];
    for curr_depth in 0..=max_depth {
        let curr_positions = all_possible_boards(&AtaxxBoard::default(), curr_depth, false);

        println!("Depth {} adds {} positions", curr_depth, curr_positions.len());

        positions.extend_from_slice(&curr_positions);
        depths.extend(std::iter::repeat(curr_depth).take(curr_positions.len()));
    }

    // let max_positions = 100;
    // drop(positions.drain(max_positions..));

    let device = Device::new(0);
    let evals = batch_tree_eval(positions.clone(), settings, zero_settings, graph, mapper, device);

    let output_path = "output.txt";
    let mut output = BufWriter::new(File::create(output_path)?);

    assert_eq!(positions.len(), evals.len());
    assert_eq!(positions.len(), depths.len());

    writeln!(&mut output, "index, depth, value, fen")?;
    for pi in 0..positions.len() {
        writeln!(
            &mut output,
            "{}, {}, {}, {:?},",
            pi,
            depths[pi],
            evals[pi].wdl.value(),
            positions[pi].to_fen()
        )?;
    }

    output.flush()?;
    Ok(())
}
