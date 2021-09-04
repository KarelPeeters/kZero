use rand::{Rng, thread_rng};

use alpha_zero::games::ataxx_cnn_network::AtaxxCNNNetwork;
use alpha_zero::non_solve_zero::{ZeroSettings, ZeroBot};
use board_game::games::ataxx::AtaxxBoard;
use board_game::util::board_gen::random_board_with_moves;
use board_game::util::bot_game;
use cuda_sys::wrapper::handle::Device;
use board_game::util::bot_game::BotGameResult;

fn main() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build_global()
        .unwrap();

    // let mut wdls = vec![];

    // for gen in [250] {
    //     println!("Comparing gen {}", gen);

        let path_l = &format!("../data/ataxx/test_loop/training/gen_652/model_1_epochs.onnx");
        let path_r = &format!("../data/ataxx/test_loop/training/gen_652/model_1_epochs.onnx");

        println!("Comparing\n{}vs\n{}", path_l, path_r);

        let iterations_l = 1000;
        let iterations_r = 1000;

        let settings_l = ZeroSettings { batch_size: 10, exploration_weight: 2.0, random_symmetries: false };
        let settings_r = ZeroSettings { batch_size: 10, exploration_weight: 2.0, random_symmetries: false };

        let result = compare_bots(
            || {
                let network = AtaxxCNNNetwork::load(path_l, settings_l.batch_size, Device::new(0));
                ZeroBot::new(iterations_l, settings_l, network, thread_rng())
            },
            || {
                let network = AtaxxCNNNetwork::load(path_r, settings_r.batch_size, Device::new(0));
                ZeroBot::new(iterations_r, settings_r, network, thread_rng())
            },
            true,
            true,
        ).unwrap();

        // wdls.push(vec![gen as f32, result.win_rate_l, result.draw_rate, result.win_rate_r, result.elo_l]);
    // }
    //
    // println!("{:?}", wdls);
}

fn compare_bots<R1: Rng, R2: Rng>(
    bot_l: impl Fn() -> ZeroBot<AtaxxBoard, AtaxxCNNNetwork, R1> + Sync,
    bot_r: impl Fn() -> ZeroBot<AtaxxBoard, AtaxxCNNNetwork, R2> + Sync,
    tree: bool,
    game: bool,
) -> Option<BotGameResult> {
    if tree {
        let board = AtaxxBoard::from_fen("1xxxxxx/xxxxxxx/xxxxxxx/ooooooo/ooooooo/ooooooo/ooooooo x 98");
        println!("{}", board);
        println!("{}", bot_l().build_tree(&board).display(1, false));
        println!("{}", bot_r().build_tree(&board).display(1, false));
    }

    if game {
        let result = bot_game::run(
            || random_board_with_moves(&AtaxxBoard::default(), 2, &mut thread_rng()),
            bot_l,
            bot_r,
            20, true, Some(1),
        );
        println!("{:#?}", result);
        Some(result)
    } else {
        None
    }
}
