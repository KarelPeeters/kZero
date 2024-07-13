use board_game::games::chess::ChessBoard;
use board_game::util::board_gen::random_board_with_moves;
use kn_graph::cpu::cpu_eval_graph;
use kn_graph::dtype::{DTensor, Tensor};
use kn_graph::onnx::load_graph_from_onnx_path;
use kz_core::mapping::chess::ChessStdMapper;
use kz_core::mapping::InputMapper;
use rand::rngs::SmallRng;
use rand::SeedableRng;

fn main() {
    let path = r#"C:\Documents\Programming\STTT\AlphaZero\data\supervised\att-again\network_72448.onnx"#;
    let graph = load_graph_from_onnx_path(path, false).unwrap();

    println!("{}", graph);

    let mut rng = SmallRng::seed_from_u64(0);
    let board = random_board_with_moves(&ChessBoard::default(), 4, &mut rng);
    let mapper = ChessStdMapper;

    let mut input = vec![];
    mapper.encode_input_full(&mut input, &board);

    let mut input_shape = vec![1];
    input_shape.extend_from_slice(&mapper.input_full_shape());

    let input = DTensor::F32(Tensor::from_shape_vec(input_shape, input).unwrap());
    let outputs = cpu_eval_graph(&graph, 1, &[input]);

    let scalars = &outputs[0];
    let policy = &outputs[1];

    println!("scalars: {:?}", scalars.unwrap_f32().unwrap().as_slice().unwrap());
    println!("policy: {:?}", policy.unwrap_f32().unwrap().as_slice().unwrap());
}
