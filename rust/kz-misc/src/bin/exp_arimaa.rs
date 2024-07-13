use board_game::board::Board;
use board_game::games::arimaa::ArimaaBoard;
use kn_cuda_sys::wrapper::handle::Device;
use kn_graph::onnx::load_graph_from_onnx_path;
use kn_graph::optimizer::optimize_graph;
use kz_core::mapping::arimaa::ArimaaSplitMapper;
use kz_core::network::cudnn::CudaNetwork;
use kz_core::zero::node::UctWeights;
use kz_core::zero::step::{FpuMode, QMode};
use kz_core::zero::wrapper::ZeroSettings;
use kz_misc::convert::pt_to_onnx::convert_pt_to_onnx;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::path::PathBuf;

fn main() {
    let device = Device::new(0);

    let path = r#"C:\Documents\Programming\STTT\kZero\data\loop\arimaa-split\4-policy-pov\training\gen_18\network.pt"#;

    let path = if path.ends_with(".pt") {
        convert_pt_to_onnx(path, "arimaa-split");
        PathBuf::from(path).with_extension("onnx")
    } else {
        PathBuf::from(path)
    };

    let graph = optimize_graph(&load_graph_from_onnx_path(path, false).unwrap(), Default::default());

    let mapper = ArimaaSplitMapper;
    let batch_size = 128;
    let visits = 1000;
    let settings = ZeroSettings::simple(batch_size, UctWeights::default(), QMode::wdl(), FpuMode::Relative(0.0));

    let mut rng = StdRng::seed_from_u64(0);
    let mut network = CudaNetwork::new(mapper, &graph, batch_size, device);

    let mut board = ArimaaBoard::default();

    while !board.is_done() {
        let tree = settings.build_tree(&board, &mut network, &mut rng, |tree| tree.root_visits() >= visits);
        println!("{}", tree.display(1, true, usize::MAX, false));

        board.play(tree.best_move().unwrap()).unwrap();

        println!("{}", board);
    }

    println!("{:?}", board.outcome());
}
