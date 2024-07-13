use itertools::Itertools;
use std::time::Instant;

use kn_graph::cpu::cpu_eval_graph_exec;
use kn_graph::dtype::{DTensor, Tensor};
// use kn_graph::dtype::{DTensor, Tensor};
use kn_graph::onnx::load_graph_from_onnx_path;
use kn_graph::optimizer::optimize_graph;

fn main() {
    let path = r#"\\192.168.0.10\Documents\Karel A0\loop\ataxx-7\16x128_gaps\training\gen_6732\network.onnx"#;
    let graph = load_graph_from_onnx_path(path, true).unwrap();
    let graph = optimize_graph(&graph, Default::default());

    let batch_size = 1;
    println!("{:?}", graph.input_shapes());

    let inputs = graph
        .input_shapes()
        .iter()
        .map(|s| DTensor::F32(Tensor::zeros(s.eval(batch_size).dims.as_slice())))
        .collect_vec();

    let start = Instant::now();
    let exec = cpu_eval_graph_exec(&graph, batch_size, &inputs, false);
    let elapsed = start.elapsed() / batch_size as u32;
    println!("{}", exec);
    println!("{:?}", elapsed);
}
