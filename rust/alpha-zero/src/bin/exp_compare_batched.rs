use rand::{SeedableRng, thread_rng};
use rand::rngs::SmallRng;

use alpha_zero::{non_solve_zero, zero};
use alpha_zero::games::ataxx_cnn_network::AtaxxCNNNetwork;
use alpha_zero::util::PanicRng;
use board_game::board::Board;
use board_game::games::ataxx::AtaxxBoard;
use cuda_sys::wrapper::handle::Device;
use board_game::util::bot_game;
use board_game::util::board_gen::random_board_with_moves;

fn main() {
    let path = "../data/ataxx/retrain_strongest/training/gen_379/model_1_epochs.onnx";

    let random = true;
    let batch_size = 500;

    let exploration_weight = 2.0;

    let mut new_network = AtaxxCNNNetwork::load(path, batch_size, Device::new(0));
    let new_settings = zero::ZeroSettings::new(batch_size, exploration_weight, random);

    let mut old_network = AtaxxCNNNetwork::load(path, batch_size, Device::new(0));
    let old_settings = non_solve_zero::ZeroSettings::new(batch_size, exploration_weight, random);

    let iterations = 36_000;
    let mut board = AtaxxBoard::default();

    println!("{}", board);
    println!("{:?}", board.outcome());

    let mut rng = thread_rng();

    println!("New:");
    println!("{}", zero::zero_build_tree(&board, iterations, new_settings, &mut new_network, &mut rng, || false).display(1, true));
    println!("Old:");
    println!("{}", non_solve_zero::zero_build_tree(&board, iterations, old_settings, &mut old_network, &mut rng, || false).display(1, true));


    rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build_global()
        .unwrap();

    println!("{:#?}", bot_game::run(
        || random_board_with_moves(&AtaxxBoard::default(), 2, &mut thread_rng()),
        || {
            let network = AtaxxCNNNetwork::load(path, batch_size, Device::new(0));
            let settings = zero::ZeroSettings::new(batch_size, exploration_weight, random);
            zero::ZeroBot::new(iterations, settings, network, thread_rng())
        },
        || {
            let network = AtaxxCNNNetwork::load(path, batch_size, Device::new(0));
            let settings = non_solve_zero::ZeroSettings::new(batch_size, exploration_weight, random);
            non_solve_zero::ZeroBot::new(iterations, settings, network, thread_rng())
        },
        100, true, Some(1),
    ));

}
