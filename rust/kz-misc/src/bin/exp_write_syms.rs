use board_game::board::{Board, BoardMoves, BoardSymmetry};
use board_game::symmetry::{D4Symmetry, Symmetry};
use board_game::wdl::OutcomeWDL;
use internal_iterator::InternalIterator;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::borrow::Cow;

use kz_core::mapping::ataxx::AtaxxStdMapper;
use kz_core::network::common::softmax_in_place;
use kz_core::network::ZeroEvaluation;
use kz_core::zero::values::ZeroValuesPov;
use kz_selfplay::binary_output::BinaryOutput;
use kz_selfplay::server::start_pos::ataxx_gen_gap_board;
use kz_selfplay::simulation::{Position, Simulation};

fn main() -> std::io::Result<()> {
    let mapper = AtaxxStdMapper::new(7);
    let mut output = BinaryOutput::new("symmetries", "ataxx-7", mapper)?;

    let mut rng = StdRng::seed_from_u64(0);
    let board = ataxx_gen_gap_board(&mut rng, mapper.size(), 0.2..=0.2);
    let played_mv = board.random_available_move(&mut rng).unwrap();

    for &sym in D4Symmetry::all() {
        let board_sym = board.map(sym);
        let played_mv_sym = board.map_move(sym, played_mv);

        println!("{:?}", sym);
        println!("{}", board_sym);

        let mut policy: Vec<f32> = board.available_moves().unwrap().map(|_| rng.gen()).collect();
        softmax_in_place(&mut policy);

        let eval = ZeroEvaluation {
            values: ZeroValuesPov::from_outcome(OutcomeWDL::Draw, 0.0),
            policy: Cow::Borrowed(&policy),
        };

        let pos = Position {
            board: board_sym.clone(),
            is_full_search: true,
            played_mv: played_mv_sym,
            zero_visits: 1000,
            zero_evaluation: eval.clone(),
            net_evaluation: eval,
        };

        let sim = Simulation {
            positions: vec![pos],
            final_board: board_sym,
        };
        output.append(&sim)?;
    }

    println!("{}", board);

    output.finish()?;
    Ok(())
}
