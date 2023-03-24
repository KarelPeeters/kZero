use nn_graph::cpu::Tensor;
use nn_graph::dot::graph_to_svg;
use nn_graph::katago::{load_katago_network, load_katago_network_path};
use nn_graph::ndarray::IxDyn;
use nn_graph::optimizer::optimize_graph;
use std::io::BufReader;

use crate::root::runner::test_all;

#[test]
fn test_katago() {
    let path = r#"C:\Documents\Programming\STTT\kZero\data\katago-ataxx7x7.bin"#;
    let graph = load_katago_network_path(path, 7).unwrap();

    // optimize
    let graph_opt = optimize_graph(&graph, Default::default());

    // render
    std::fs::create_dir_all("../ignored/graphs").unwrap();
    graph_to_svg("../ignored/graphs/before.svg", &graph, true).unwrap();
    graph_to_svg("../ignored/graphs/after.svg", &graph_opt, true).unwrap();

    // test graph and executors
    let batch_size = 16;
    let inputs = [
        Tensor::zeros(IxDyn(&[batch_size, 22, 7, 7])),
        Tensor::zeros(IxDyn(&[batch_size, 19])),
    ];
    test_all(&graph, batch_size, &inputs, None);
}
