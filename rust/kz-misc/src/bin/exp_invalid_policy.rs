use board_game::board::{Board, BoardMoves};
use board_game::games::chess::ChessBoard;
use board_game::util::board_gen::RandomBoardIterator;
use crossterm::style::available_color_count;
use internal_iterator::InternalIterator;
use itertools::Itertools;
use kn_cuda_eval::executor::CudaExecutor;
use kn_cuda_sys::wrapper::handle::Device;
use kn_graph::dtype::{DTensor, Tensor};
use kn_graph::onnx::load_graph_from_onnx_path;
use kn_graph::optimizer::{optimize_graph, OptimizerSettings};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;

use kz_core::mapping::chess::ChessStdMapper;
use kz_core::mapping::{InputMapper, PolicyMapper};
use kz_core::network::common::softmax;

fn main() {
    // let path = r"C:\Documents\Programming\STTT\kZero\data\networks\chess_16x128_gen3634.onnx";
    let path = r"\\192.168.0.10\Documents\Karel A0\loop\chess\16x128\initial_network.onnx";
    let mapper = ChessStdMapper;
    let graph = optimize_graph(
        &load_graph_from_onnx_path(path, false).unwrap(),
        OptimizerSettings::default(),
    );

    let mut network = CudaExecutor::new(Device::new(0), &graph, 1);

    let start = ChessBoard::default();
    let rng = SmallRng::seed_from_u64(0);

    let count = 100_000;

    let mut result_move_counts = vec![];
    let mut result_valid_logits = vec![];
    let mut result_invalid_logits = vec![];
    let mut result_valid_mass = vec![];

    for (board_index, board) in RandomBoardIterator::new(start, rng).unwrap().enumerate() {
        if board.is_done() {
            continue;
        }
        if board_index > count {
            break;
        }
        if board_index % 1000 == 0 {
            println!(
                "progrss {}/{} = {}",
                board_index,
                count,
                board_index as f32 / count as f32
            );
        }

        // println!("{}", board);

        let mut input = vec![];
        mapper.encode_input_full(&mut input, &board);

        let mut input_shape = vec![1];
        input_shape.extend_from_slice(&mapper.input_full_shape());
        let inputs = [DTensor::F32(Tensor::from_shape_vec(input_shape, input).unwrap())];

        let outputs = network.evaluate(&inputs);

        assert_eq!(outputs.len(), 2);
        let _scalars = outputs[0].unwrap_f32().unwrap().iter().copied().collect_vec();

        let logits_all = outputs[1].unwrap_f32().unwrap().iter().copied().collect_vec();
        let policy_all = softmax(&logits_all);

        // println!("all logits: {:?}", policy_logits);
        let moves = board.available_moves().unwrap();
        let logits_valid: Vec<_> = moves
            .clone()
            .map(|mv| logits_all[mapper.move_to_index(&board, mv)])
            .collect();
        let policy_valid = softmax(&logits_valid);

        let all_valid_mass = moves
            .clone()
            .map(|mv| {
                let pi = mapper.move_to_index(&board, mv);
                policy_all[pi]
            })
            .collect::<Vec<_>>();
        let all_valid_mass = all_valid_mass.iter().sum::<f32>();

        // board.available_moves().unwrap().enumerate().for_each(|(mi, mv)| {
        //     let pi = mapper.move_to_index(&board, mv);
        //     println!("  {}: {} {}", mv, logits_all[pi], policy_valid[mi])
        // });

        let average_valid_logit = logits_valid.iter().sum::<f32>() / logits_valid.len() as f32;
        let average_invalid_logit = logits_all.iter().sum::<f32>() / logits_all.len() as f32;

        // println!("  average valid logit: {}", average_valid_logit);
        // println!("  average invalid logit: {}", average_invalid_logit);
        // println!("  all valid mass: {}", all_valid_mass);

        result_move_counts.push(moves.count());
        result_valid_logits.push(average_valid_logit);
        result_invalid_logits.push(average_invalid_logit);
        result_valid_mass.push(all_valid_mass);
    }

    let mut file = File::create("output_untrained.txt").unwrap();

    writeln!(file, "move_counts = {:?}", result_move_counts).unwrap();
    writeln!(file, "valid_logits = {:?}", result_valid_logits).unwrap();
    writeln!(file, "invalid_logits = {:?}", result_invalid_logits).unwrap();
    writeln!(file, "valid_mass = {:?}", result_valid_mass).unwrap();

    file.flush().unwrap();
}
