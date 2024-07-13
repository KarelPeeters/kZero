use board_game::ai::minimax::MiniMaxBot;
use board_game::ai::simple::RandomBot;
use board_game::games::ataxx::AtaxxBoard;
use board_game::heuristic::ataxx::AtaxxTileHeuristic;
use board_game::util::board_gen::random_board_with_moves;
use itertools::Itertools;
use rand::rngs::StdRng;
use rand::{thread_rng, SeedableRng};

use kz_core::bot::WrapAsync;
use kz_misc::eval::tournament::{box_bot, run_tournament};

fn main() {
    let bots = vec![
        ("random", box_bot(|| WrapAsync(RandomBot::new(StdRng::from_entropy())))),
        (
            "mm-1",
            box_bot(|| {
                WrapAsync(MiniMaxBot::new(
                    1,
                    AtaxxTileHeuristic::default(),
                    StdRng::from_entropy(),
                ))
            }),
        ),
        (
            "mm-3",
            box_bot(|| {
                WrapAsync(MiniMaxBot::new(
                    3,
                    AtaxxTileHeuristic::default(),
                    StdRng::from_entropy(),
                ))
            }),
        ),
        (
            "mm-5",
            box_bot(|| {
                WrapAsync(MiniMaxBot::new(
                    5,
                    AtaxxTileHeuristic::default(),
                    StdRng::from_entropy(),
                ))
            }),
        ),
        (
            "mm-7",
            box_bot(|| {
                WrapAsync(MiniMaxBot::new(
                    7,
                    AtaxxTileHeuristic::default(),
                    StdRng::from_entropy(),
                ))
            }),
        ),
    ];

    let mut rng = thread_rng();
    let positions = (0..16)
        .map(|_| random_board_with_moves(&AtaxxBoard::diagonal(7), 4, &mut rng))
        .collect_vec();

    let result = run_tournament(bots, positions, Some(6), true, true, || {});

    println!("{}", result);
}
