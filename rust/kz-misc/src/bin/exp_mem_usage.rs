use itertools::Itertools;
use kn_cuda_eval::executor::CudaExecutor;
use kn_cuda_sys::wrapper::handle::Device;
use kn_graph::onnx::load_graph_from_onnx_path;
use kn_graph::optimizer::optimize_graph;

fn main() {
    let graph = optimize_graph(
        &load_graph_from_onnx_path(
            "C:/Documents/Programming/STTT/AlphaZero/data/networks/chess_16x128_gen3634.onnx",
            false,
        )
        .unwrap(),
        Default::default(),
    );

    let device = Device::new(0);

    let mem_usages = std::iter::once(1024)
        .map(|batch_size| {
            let bytes = CudaExecutor::new(device, &graph, batch_size).mem_usage.shared_bytes;
            println!("{} -> {}", batch_size, bytes);
            bytes
        })
        .collect_vec();

    println!("{:?}", mem_usages);
}
