use board_game::board::{Board, BoardMoves};
use board_game::games::chess::{chess_game_to_pgn, ChessBoard};
use decorum::N32;
use internal_iterator::InternalIterator;
use itertools::Itertools;

use kn_cuda_sys::wrapper::handle::Device;
use kn_graph::graph::BinaryOp;
use kn_graph::graph::Operation::Binary;
use kn_graph::onnx::load_graph_from_onnx_path;
use kn_graph::optimizer::{optimize_graph, OptimizerSettings};
use kz_core::mapping::chess::ChessStdMapper;
use kz_core::network::cudnn::CudaNetwork;
use kz_core::network::{Network, ZeroEvaluation};
use kz_selfplay::binary_output::BinaryOutput;
use kz_selfplay::simulation::{Position, Simulation};

fn main() {
    let path = r"C:\Documents\Programming\STTT\kZero\data\networks\chess_16x128_gen3634.onnx";
    let graph = load_graph_from_onnx_path(path, false).unwrap();
    let mapper = ChessStdMapper;
    let mut network = CudaNetwork::new(
        mapper,
        &optimize_graph(&graph, OptimizerSettings::default()),
        1,
        Device::new(0),
    );

    let start = ChessBoard::default();
    let mut board = start.clone();
    let mut all_moves = vec![];

    let mut positions = vec![];

    while let Ok(moves) = board.available_moves() {
        let eval = network.evaluate(&board);
        let mv_index = eval.policy.iter().map(|&f| N32::from(f)).position_max().unwrap();
        let mv = moves.nth(mv_index).unwrap();
        all_moves.push(mv);
        println!("{}", mv);

        positions.push(Position {
            board: board.clone(),
            is_full_search: false,
            played_mv: mv,
            zero_visits: 0,
            zero_evaluation: ZeroEvaluation::nan(eval.policy.len()),
            net_evaluation: eval,
        });

        board.play(mv).unwrap();
    }

    let name = "kZero 16x128_gen3634 1Node";
    println!("{}", chess_game_to_pgn(name, name, &start, &all_moves));

    let sim = Simulation {
        positions,
        final_board: board,
    };
    let mut output = BinaryOutput::new("ignored/max_policy_game", "chess", mapper).unwrap();
    output.append(&sim).unwrap();
    output.finish().unwrap();
}
