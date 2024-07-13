use std::fs::File;
use std::io::{stdin, stdout, Write};
use std::time::Instant;

use board_game::board::{Board, BoardDone};
use board_game::games::go::GoBoard;
use board_game::interface::gtp::engine::{Action, GtpBot, GtpEngineState, TimeInfo};
use rand::rngs::SmallRng;
use rand::SeedableRng;

use kn_cuda_sys::wrapper::handle::Device;
use kn_graph::onnx::load_graph_from_onnx_path;
use kn_graph::optimizer::optimize_graph;
use kz_core::mapping::go::GoStdMapper;
use kz_core::network::cudnn::CudaNetwork;
use kz_core::network::Network;
use kz_core::zero::node::UctWeights;
use kz_core::zero::step::{FpuMode, QMode};
use kz_core::zero::wrapper::ZeroSettings;

fn main() {
    let path = r#"C:\Documents\Programming\STTT\kZero\data\networks\go9_first_gen716.onnx"#;
    let mapper = GoStdMapper::new(9, false);

    let batch_size = 64;
    let settings = ZeroSettings::simple(batch_size, UctWeights::default(), QMode::wdl(), FpuMode::Relative(0.0));

    let graph = optimize_graph(&load_graph_from_onnx_path(&path, false).unwrap(), Default::default());
    let network = CudaNetwork::new(mapper, &graph, batch_size, Device::new(0));

    let engine = ZeroGtpBot {
        settings,
        network,
        resign_threshold: -0.9999,
    };

    let log = File::create("kzero_log.txt").unwrap();
    GtpEngineState::new()
        .run_loop(engine, stdin().lock(), stdout().lock(), log)
        .unwrap();
}

struct ZeroGtpBot<N: Network<GoBoard>> {
    settings: ZeroSettings,
    network: N,
    resign_threshold: f32,
}

impl<N: Network<GoBoard>> GtpBot for ZeroGtpBot<N> {
    fn select_action(&mut self, board: &GoBoard, time: &TimeInfo, log: &mut impl Write) -> Result<Action, BoardDone> {
        assert!(
            self.resign_threshold < 0.0,
            "Resign threshold must be negative, got {}",
            self.resign_threshold
        );

        let mut rng = SmallRng::from_entropy();
        let random_mv = board.random_available_move(&mut rng)?;

        let start = Instant::now();
        // TODO what expected stones left to use here?
        let time_to_use = time.simple_time_to_use(50.0);

        let mut prev_time = start;

        writeln!(log, "planning to use {}s", time_to_use).unwrap();
        let tree = self.settings.build_tree(board, &mut self.network, &mut rng, |_| {
            let time = Instant::now();
            let step_elapsed = (time - prev_time).as_secs_f32();
            prev_time = time;

            // would the next step risk exceeding over the time limit?
            start.elapsed().as_secs_f32() + step_elapsed * 1.5 >= time_to_use
        });
        writeln!(log, "actually used {}s", start.elapsed().as_secs_f32()).unwrap();

        writeln!(log, "{}", tree.display(1, true, usize::MAX, false)).unwrap();

        let action = match tree.best_move() {
            None => Action::Move(random_mv),
            Some(mv) => {
                if tree.values().wdl.value() < self.resign_threshold {
                    Action::Resign
                } else {
                    Action::Move(mv)
                }
            }
        };

        writeln!(log, "selected action: {:?}", action).unwrap();

        Ok(action)
    }
}
