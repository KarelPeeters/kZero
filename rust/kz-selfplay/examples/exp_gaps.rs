use rand::rngs::StdRng;
use rand::SeedableRng;

use kz_selfplay::server::start_pos::ataxx_start_pos;

fn main() {
    let size = 7;
    let start_pos = ataxx_start_pos(size, "default");

    let mut rng = StdRng::from_entropy();

    for _ in 0..100 {
        println!("{}", start_pos(&mut rng));
    }
}
