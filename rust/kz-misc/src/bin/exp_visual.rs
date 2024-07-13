use board_game::board::Board;
use board_game::games::chess::ChessBoard;
use rand::thread_rng;

use kn_graph::onnx::load_graph_from_onnx_path;
use kz_core::mapping::chess::ChessStdMapper;
use kz_core::network::cpu::CPUNetwork;
use kz_misc::visualize::visualize_network_activations_split;

fn main() {
    let path = "C:/Documents/Programming/STTT/AlphaZero/data/loop/chess/real/training/gen_386/network.onnx";
    let mapper = ChessStdMapper;

    println!("Generating boards");
    let mut rng = thread_rng();

    let start = ChessBoard::default();
    let mut boards = vec![];

    while boards.len() < 128 {
        let mut board = start.clone();
        while !board.is_done() {
            boards.push(board.clone());
            board.play_random_available_move(&mut rng).unwrap();
        }
    }

    println!("Total boards: {}", boards.len());
    println!("Loading network");
    let graph = load_graph_from_onnx_path(path, false).unwrap();
    println!("{}", graph);
    let mut network = CPUNetwork::new(mapper, graph.clone());

    println!("Calculating and rendering images");
    let (images_a, images_b) = visualize_network_activations_split(&mut network, &boards, None, false, true);

    println!("Saving images");
    std::fs::create_dir_all("ignored/visualize").unwrap();
    for (i, image) in images_a.iter().enumerate() {
        image.save(format!("ignored/visualize/board_a_{}.png", i)).unwrap();
    }
    for (i, image) in images_b.iter().enumerate() {
        image.save(format!("ignored/visualize/board_b_{}.png", i)).unwrap();
    }
}
