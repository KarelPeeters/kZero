use rand::thread_rng;

use alpha_zero::games::ataxx_cnn_network::AtaxxCNNNetwork;
use alpha_zero::zero::{zero_build_tree, ZeroSettings};
use board_game::games::ataxx::AtaxxBoard;
use cuda_sys::wrapper::handle::Device;

fn main() {
    let device = Device::new(0);

    let mut rng = thread_rng();

    let iterations = 10_000;
    let settings = ZeroSettings::new(4.0, true);

    let paths = [
        "../data/derp/good_test_loop/gen_40/model_1_epochs.onnx",
    ];

    let board = AtaxxBoard::from_fen("xx4x/7/7/7/7/o6/oo4o");
    println!("{}", board);
    // board.play(Move::Jump { from: Coord::from_xy(1, 6), to: Coord::from_xy(3, 4) });
    // println!("{}", board);

    for &path in &paths {
        println!("{}:", path);
        let mut network = AtaxxCNNNetwork::load(path, 1, device);
        let tree = zero_build_tree(&board, iterations, settings, &mut network, &mut rng, || false);
        println!("{}", tree.display(1));
    }
}