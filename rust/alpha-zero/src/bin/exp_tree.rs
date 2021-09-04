use rand::thread_rng;

use alpha_zero::games::ataxx_cnn_network::AtaxxCNNNetwork;
use alpha_zero::zero::{ZeroBot, ZeroSettings};
use board_game::games::ataxx::AtaxxBoard;
use board_game::util::board_gen::{random_board_with_forced_win};
use cuda_sys::wrapper::handle::Device;

fn main() {
    let device = Device::new(0);
    let path = "../data/ataxx/test_loop/training/gen_360/model_1_epochs.onnx";
    let iterations = 1_000_000;
    let batch_size = 100;
    let settings = ZeroSettings::new(batch_size, 2.0, false);
    let network = AtaxxCNNNetwork::load(path, batch_size, device);
    let mut zero_bot = ZeroBot::new(iterations, settings, network, thread_rng());

    // let board = random_board_with_forced_win(&AtaxxBoard::new_without_gaps(), 6, &mut thread_rng());
    let board = AtaxxBoard::from_fen("xo5/7/7/7/7/7/7");
    println!("{}", board);

    let tree = zero_bot.build_tree(&board);
    println!("{}", tree.display(1, false));
    // println!("{}", tree.display(2));
}
