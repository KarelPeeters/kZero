use board_game::arimaa_engine_step::Action;
// use board_game::arimaa_engine_step::full_move::convert_move_string_to_actions;
use board_game::board::Board;
use board_game::games::arimaa::ArimaaBoard;
use kz_core::network::dummy::DummyNetwork;
use kz_core::zero::node::UctWeights;
use kz_core::zero::step::{FpuMode, QMode};
use kz_core::zero::wrapper::ZeroSettings;
use rand::rngs::SmallRng;
use rand::SeedableRng;

fn main() {
    let game = std::fs::read_to_string("ignored/game.txt").unwrap();

    let mut actions: Vec<Action> = vec![];
    for line in game.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("#") {
            break;
        }
        // TODO uncomment this
        // actions.extend_from_slice(&convert_move_string_to_actions(line));
    }

    let visits = 1_000_000;
    let mut network = DummyNetwork;
    let mut rng = SmallRng::seed_from_u64(0);

    let mut board = ArimaaBoard::default();
    let mut trees = false;

    for mv in actions {
        println!("{}", board);
        println!("{:?}", board.next_player());

        if mv.to_string() == "b5n" {
            trees = true;
        }

        if trees {
            let settings = ZeroSettings::simple(1, UctWeights::default(), QMode::wdl(), FpuMode::Relative(0.0));
            let tree = settings.build_tree(&board, &mut network, &mut rng, |tree| tree.root_visits() >= visits);
            println!("{}", tree.display(1, true, usize::MAX, false));
        }

        println!("Playing mv {}", mv);
        board.play(mv).unwrap();
    }

    println!("{:?}", board);
    println!("{:?}", board.outcome());
}
