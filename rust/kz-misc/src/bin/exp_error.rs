use std::path::PathBuf;

use board_game::games::chess::ChessBoard;

use kn_cuda_eval::tester::{assert_tensors_match, eval_cudnn, load_check_data};
use kn_graph::cpu::cpu_eval_graph;
use kn_graph::onnx::load_graph_from_onnx_path;
use kn_graph::optimizer::{optimize_graph, OptimizerSettings};
use kz_core::mapping::chess::ChessStdMapper;
use kz_core::mapping::PolicyMapper;

fn main() {
    let board = ChessBoard::default();
    let mv = ChessStdMapper.index_to_move(&board, 596);
    println!("{}", mv.unwrap());

    let path = "C:/Documents/Programming/STTT/AlphaZero/data/loop/chess/real-128/training/gen_157/network.onnx";

    let path_bin = PathBuf::from(path).with_extension("bin");

    let loaded_graph = load_graph_from_onnx_path(path, false).unwrap();
    let graph = optimize_graph(&loaded_graph, OptimizerSettings::default());

    let check_data = std::fs::read(path_bin).expect("Failed to read check data");

    let (batch_size, inputs, expected_outputs) = load_check_data(&graph, &check_data);

    let outputs_cpu = cpu_eval_graph(&graph, batch_size, &inputs);
    let outputs_cudnn = eval_cudnn(&graph, batch_size, &inputs, false);

    println!("Expected vs CPU");
    assert_tensors_match(&expected_outputs, &outputs_cpu, true);
    println!("Expected vs GPU");
    assert_tensors_match(&expected_outputs, &outputs_cudnn, true);
}
