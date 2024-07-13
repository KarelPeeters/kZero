use board_game::arimaa_engine_step::Action;
use std::path::PathBuf;
use std::str::FromStr;
use std::thread::Builder;

// use board_game::arimaa_engine_step::full_move::{convert_actions_to_move_string, convert_move_string_to_actions};
use board_game::board::{Board, PlayError};
use board_game::games::arimaa::ArimaaBoard;
use board_game::interface::aei::{Command, IdType, OptionName, Response, TCOptionName};
use rand::rngs::StdRng;
use rand::SeedableRng;

use kn_cuda_sys::wrapper::handle::Device;
use kn_graph::onnx::load_graph_from_onnx_path;
use kn_graph::optimizer::optimize_graph;
use kz_core::mapping::arimaa::ArimaaSplitMapper;
use kz_core::network::job_channel::job_pair;
use kz_misc::convert::pt_to_onnx::convert_pt_to_onnx;
use kz_selfplay::server::executor::{alphazero_batched_executor_loop, RunCondition};

fn main() {
    let path = r#"C:\Documents\Programming\STTT\kZero\data\loop\arimaa-split\8-always-temp\gen_45\network.pt"#;
    convert_pt_to_onnx(&path, "arimaa-split");
    let graph = optimize_graph(
        &load_graph_from_onnx_path(PathBuf::from(path).with_extension("onnx"), false).unwrap(),
        Default::default(),
    );

    let batch_size = 128;
    let device = Device::new(0);
    let mapper = ArimaaSplitMapper;

    let (_client, server) = job_pair(1);
    // let (sender, receiver) = flume::bounded(0);

    Builder::new()
        .name("executor".to_owned())
        .spawn(move || alphazero_batched_executor_loop(batch_size, device, mapper, RunCondition::Any, graph, server))
        .unwrap();

    Builder::new()
        .name("searcher".to_owned())
        .spawn(move || {
            // main_searcher();
        })
        .unwrap();

    main_io();
}

// fn main_searcher(client: EvalClient<ArimaaBoard>, receiver: ) {
//
// }

fn main_io() {
    let mut board = ArimaaBoard::default();
    let mut _time_per_move = 0;
    let mut rng = StdRng::from_entropy();

    let mut line = String::new();
    loop {
        line.clear();
        std::io::stdin().read_line(&mut line).unwrap();
        let line = line.trim();

        let command = match Command::parse(line) {
            Ok(command) => command,
            Err(_) => {
                println!(
                    "{}",
                    Response::Log(format!("Error: failed to parse command '{}'", line))
                );
                continue;
            }
        };

        match command {
            Command::AEI => {
                println!("{}", Response::ProtocolV1);
                respond_id();
                println!("{}", Response::AeiOk);
            }
            Command::IsReady => {
                println!("{}", Response::ReadyOk);
            }
            Command::NewGame => {
                board = ArimaaBoard::default();
                //TODO reset used time counters, keep settings
            }
            Command::SetPosition(position) => match ArimaaBoard::from_str(&position) {
                Ok(new_board) => {
                    board = new_board;
                }
                Err(_) => println!(
                    "{}",
                    Response::Log(format!("Error: error while parsing board '{}'", position))
                ),
            },
            Command::SetOption { name, value } => {
                match name {
                    OptionName::TC(TCOptionName::TcMove) => {
                        _time_per_move = value.unwrap().parse::<u32>().unwrap();
                    }
                    _ => {}
                }

                // TODO support at least TC options
            }
            Command::MakeMove(full_move) => {
                // let actions = convert_move_string_to_actions(&full_move);
                let actions: Vec<Action> = vec![];

                let mut next_board = board.clone();
                let mut failed = false;

                for action in actions {
                    match next_board.play(action) {
                        Ok(()) => {}
                        Err(e) => {
                            let msg = match e {
                                PlayError::BoardDone => format!("Board done, can't play '{}'", action),
                                PlayError::UnavailableMove => format!("Action not available: '{}'", action),
                            };

                            println!("{}", Response::Log(msg));
                            failed = true;
                            break;
                        }
                    }

                    if next_board.play(action).is_err() {
                        println!("{}", Response::Log(format!("Action '{}' not available", action)));
                        failed = true;
                        break;
                    }
                }

                if !failed {
                    board = next_board;
                }
            }
            Command::Go { ponder } => {
                // TODO implement ponder

                if board.is_done() {
                    println!("{}", Response::Log(format!("Board is already done")));
                    continue;
                }

                if ponder {
                    continue;
                }

                let mut next_board = board.clone();
                let start_player = next_board.next_player();
                let mut actions = vec![];

                loop {
                    let action = next_board.random_available_move(&mut rng).unwrap();
                    next_board.play(action).unwrap();
                    actions.push(action);

                    if next_board.next_player() != start_player {
                        break;
                    }
                }

                // let full_mv = convert_actions_to_move_string(board.state().clone(), &actions);
                // println!("{}", Response::BestMove(full_mv));
            }
            Command::Stop => {
                // TODO stop pondering once we implement that
            }
            Command::Quit => break,
        }
    }
}

fn respond_id() {
    let responses = [
        (IdType::Name, "kZero"),
        (IdType::Author, "KarelPeeters"),
        (IdType::Version, "0.0.1"),
    ];

    for (ty, value) in responses {
        let value = value.to_owned();
        let response = Response::Id { ty, value };
        println!("{}", response);
    }
}
