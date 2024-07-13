use board_game::games::ataxx::AtaxxBoard;
use kn_cuda_sys::wrapper::handle::Device;
use kn_graph::onnx::load_graph_from_onnx_path;
use kn_graph::optimizer::{optimize_graph, OptimizerSettings};
use kz_core::mapping::ataxx::AtaxxStdMapper;
use kz_core::network::cudnn::CudaNetwork;
use kz_core::network::multibatch::MultiBatchNetwork;
use kz_core::network::Network;

fn main() {
    let network_path =
        r#"C:\Documents\Programming\STTT\kZero\data\loop\ataxx-7\16-128-v2\tmp\curr_network\network_5012.onnx"#;
    let graph = optimize_graph(
        &load_graph_from_onnx_path(network_path, false).unwrap(),
        OptimizerSettings::default(),
    );

    let mut network = MultiBatchNetwork::build_sizes(&[16, 32, 64, 128], |batch_size| {
        CudaNetwork::new(AtaxxStdMapper::new(7), &graph, batch_size, Device::new(0))
    });

    for i in 0..=128 {
        println!("Evaluating {} boards", i);
        let boards = vec![AtaxxBoard::default(); i];
        let result = network.evaluate_batch(&boards);
        assert_eq!(result.len(), boards.len());
    }
}
