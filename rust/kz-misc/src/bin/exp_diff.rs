use board_game::board::Board;
use board_game::games::ataxx::AtaxxBoard;
use rand::rngs::StdRng;
use rand::SeedableRng;

use kn_cuda_sys::wrapper::handle::Device;
use kn_graph::onnx::load_graph_from_onnx_path;
use kn_graph::optimizer::optimize_graph;
use kz_core::mapping::ataxx::AtaxxStdMapper;
use kz_core::network::cudnn::CudaNetwork;
use kz_core::network::Network;

fn main() {
    let path = r#"C:\Documents\Programming\STTT\kZero\data\networks\ataxx_16x128_gen_1364_old.onnx"#;
    let graph = load_graph_from_onnx_path(path, false).unwrap();

    println!("Unoptimized");
    println!("{}", graph);

    let graph = optimize_graph(&graph, Default::default());

    println!("Optimized");
    println!("{}", graph);

    // for v in graph.values() {
    //     let info = &graph[v];
    //     if let Operation::Constant { data } = &info.operation {
    //         println!("{:?} {:?}", info.shape, data.0);
    //     }
    // }

    let mapper = AtaxxStdMapper::new(7);
    let mut network = CudaNetwork::new(mapper, &graph, 1, Device::new(0));

    // println!("{:?}", network.executor());

    let mut rng = StdRng::seed_from_u64(0);
    let mut i = 0;

    loop {
        let mut board = AtaxxBoard::default();
        while !board.is_done() {
            let eval = network.evaluate(&board);
            println!("{}: {:?}", i, eval);

            board.play_random_available_move(&mut rng).unwrap();
            i += 1;

            if i > 1000 {
                return;
            }
        }
    }
}
