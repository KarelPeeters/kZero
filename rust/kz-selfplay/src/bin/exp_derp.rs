use kz_selfplay::server::start_pos::{ataxx_gen_gap_board, ataxx_start_pos};
use rand::rngs::{SmallRng, StdRng};
use rand::SeedableRng;

fn main() {
    let mut rng = StdRng::seed_from_u64(0);
    let start_pos = ataxx_start_pos(7, "random-gaps-v1");

    for _ in 0..100 {
        let pos = start_pos(&mut rng);
        println!("{}", pos);
    }
}
