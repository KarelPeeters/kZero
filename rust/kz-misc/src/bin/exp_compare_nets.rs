#![allow(dead_code)]

use board_game::board::Board;
use board_game::games::ataxx::AtaxxBoard;
use board_game::util::board_gen::random_board_with_moves;
use board_game::util::bot_game;
use rand::rngs::StdRng;
use rand::{thread_rng, Rng, SeedableRng};
use rayon::ThreadPoolBuilder;

use kn_cuda_eval::Device;
use kn_graph::onnx::load_graph_from_onnx_path;
use kn_graph::optimizer::{optimize_graph, OptimizerSettings};
use kz_core::mapping::ataxx::AtaxxStdMapper;
use kz_core::mapping::BoardMapper;
use kz_core::network::cudnn::CudaNetwork;
use kz_core::zero::node::UctWeights;
use kz_core::zero::step::{FpuMode, QMode};
use kz_core::zero::tree::Tree;
use kz_core::zero::wrapper::{ZeroBot, ZeroSettings};

// 50 vs 80:                                    WDL { win: 24, draw: 8, loss: 49 }
// 80 vs 115:                                   WDL { win: 45, draw: 10, loss: 40 }

// initial-28 vs smaller-buffer-28:             WDL { win: 22, draw: 4, loss: 24 }
// initial-38 vs smaller-buffer-38:             WDL { win: 20, draw: 7, loss: 23 }

// initial-90 vs small-eps-90:                  WDL { win: 23, draw: 5, loss: 22 }
// initial-90 vs less-visits-90:                WDL { win: 18, draw: 12, loss: 20 }
// initial-90 vs less-visits-150:               WDL { win: 21, draw: 8, loss: 21 }

// less-visits-150 vs less-visits-deeper-150              WDL { win: 24, draw: 6, loss: 20 }
// less-visits-200 vs less-visits-more-exploration-200    WDL { win: 24, draw: 6, loss: 11 } (rekt lol)
// less-visits-80 vs more-exploration-80                  WDL { win: 10, draw: 1, loss: 4 }

// 16x128 gen_2128 vs 8x64 gen gen_1859

// 256/2   vs    256:    WDL { win: 7, draw: 14, loss: 19 }
// 256     vs    256:    WDL { win: 24, draw: 11, loss: 5 }
// 1       vs      1:    WDL { win: 21, draw: 10, loss: 9 }
// 4096    vs   4096:
// 4096/2  vs   4096:

// 3338 vs 3071 @ 4096:  WDL { win: 21, draw: 10, loss: 9 }

// 8x64 PolicyReg vs Parent -> bad tests, they always use the same starting position!
// @   32:    WDL { win: 0,  draw: 0,  loss: 40 }
// @  128:    WDL { win: 0,  draw: 20, loss: 20 }
// @  256:    WDL { win: 20, draw: 0,  loss: 20 }
// @ 4096:    WDL { win: 0,  draw: 0,  loss: 20 }

// tmp\network_661.onnx vs ataxx_16x128_gen_1364.onnx
// WDL { win: 32, draw: 4, loss: 134 }
// tmp\network_1122.onnx vs ...
// WDL { win: 79, draw: 10, loss: 111 }
// tmp\network_1978.onxx vs ...
// WDL { win: 110, draw: 9, loss: 81 } elo +50.7
// tmp\network_2049.onnx vs ...
// WDL { win: 110, draw: 4, loss: 86 }
// tmp\network_2383.onnx vs ...
// WDL { win: 115, draw: 9, loss: 76 } elo +68.6
// tmp\network_2743.onnx vs ...
// WDL { win: 122, draw: 11, loss: 67 } elo +98.1
// tmp\network_2812.onnx vs ...
// WDL { win: 130, draw: 10, loss: 60 } elo +127.0
// tmp\network_2872.onnx vs ...
// WDL { win: 120, draw: 7, loss: 73 } elo +83.2
// same at 2048 nodes (all previous at 512 nodes)
// WDL { win: 115, draw: 15, loss: 70 } elo +79.5
// 2048 nodes, tmp\network_3290.onnx vs ...
// WDL { win: 125, draw: 17, loss: 58 } elo +121.1
// WDL { win: 106, draw: 24, loss: 70 } elo + 63.2
// WDL { win: 25, draw: 2, loss: 20 }

fn main() {
    let path_l =
        r#"C:\Documents\Programming\STTT\kZero\data\loop\ataxx-7\16-128-v2\tmp\curr_network\network_4460.onnx"#;
    let path_r = r#"C:\Documents\Programming\STTT\kZero\data\networks\ataxx_16x128_gen_1364.onnx"#;

    let batch_size = 64;
    let settings_l = ZeroSettings::simple(batch_size, UctWeights::default(), QMode::wdl(), FpuMode::Relative(0.0));
    let settings_r = ZeroSettings::simple(batch_size, UctWeights::default(), QMode::wdl(), FpuMode::Relative(0.0));
    let visits_l = 2048;
    let visits_r = 2048;

    println!("{}", path_l);
    println!("{:?}", settings_l);
    println!("vs");
    println!("{}", path_r);
    println!("{:?}", settings_r);

    let mapper = AtaxxStdMapper::new(7);

    ThreadPoolBuilder::new().num_threads(4).build_global().unwrap();
    let result = bot_game::run(
        || {
            let mut rng = thread_rng();
            random_board_with_moves(&AtaxxBoard::default(), rng.gen_range(2..5), &mut rng)
        },
        || {
            ZeroBot::new(
                network(mapper, batch_size, path_l),
                settings_l,
                visits_l,
                StdRng::from_entropy(),
            )
        },
        || {
            ZeroBot::new(
                network(mapper, batch_size, path_r),
                settings_r,
                visits_r,
                StdRng::from_entropy(),
            )
        },
        100,
        true,
        |wdl, _| {
            println!("{:?}", wdl);

            // let pgn = replay.to_pgn();
            // println!("{}", pgn);
            // println!("{{ current: {:?} }}", wdl);
            // println!("\n");
        },
    );
    println!("{:?}", result);

    /*
    let mut bot_l = ZeroBot::new(network(mapper, batch_size, path_l), settings_l, visits_l);
    let mut bot_r = ZeroBot::new(network(mapper, batch_size, path_r), settings_r, visits_r);

    let mut board = ChessBoard::default();
    while !board.is_done() {
        let tree_l = bot_l.build_tree(&board);
        let tree_r = bot_r.build_tree(&board);

        let mv_l = tree_l.best_move().unwrap();
        let mv_r = tree_r.best_move().unwrap();

        if mv_l != mv_r {
            println!("!!!");
        }
        println!("{}", board);
        println!(
            "{} @ ({}, {}) vs {} @ ({}, {})",
            mv_l, visits_for_move(&tree_l, mv_l), visits_for_move(&tree_r, mv_l),
            mv_r, visits_for_move(&tree_l, mv_r), visits_for_move(&tree_r, mv_r),
        );
        println!();

        board.play(mv_r);
    }
     */
}

fn visits_for_move<B: Board>(tree: &Tree<B>, mv: B::Move) -> u64 {
    let child = tree[0]
        .children
        .unwrap()
        .iter()
        .find(|&c| tree[c].last_move == Some(mv))
        .unwrap();
    tree[child].complete_visits
}

fn network<B: Board, M: BoardMapper<B>>(mapper: M, batch_size: usize, path: &str) -> CudaNetwork<B, M> {
    let graph = optimize_graph(
        &load_graph_from_onnx_path(path, false).unwrap(),
        OptimizerSettings::default(),
    );
    CudaNetwork::new(mapper, &graph, batch_size, Device::new(0))
}
