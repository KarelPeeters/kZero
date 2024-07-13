// use std::time::Instant;
//
// use kn_cuda_sys::wrapper::handle::Device;
//
// use kz_core::mapping::chess::ChessStdMapper;
// use kz_core::network::muzero::MuZeroGraphs;
//
// fn main() {
//     let path =
//         r#"C:\Documents\Programming\STTT\AlphaZero\data\loop_mu\chess\profile\tmp\curr_network\network_initial_"#;
//     let device = Device::new(0);
//     let batch_size = 2048;
//     let mapper = ChessStdMapper;
//
//     println!("Loading graphs");
//     let graphs = MuZeroGraphs::load(&path, mapper).unwrap();
//     let graphs = graphs.optimize(Default::default());
//     let fused = graphs.fuse(Default::default());
//     // let state_saved_size = fused.info.state_saved_shape(mapper).size().eval(1);
//
//     println!("Building executors");
//     let mut exec0 = fused.expand_executor(device, batch_size);
//
//     let iterations = 100;
//     let start = Instant::now();
//
//     unsafe {
//         println!("Queuing operations");
//         for i in 0..iterations {
//             if i == iterations - 1 {
//                 exec0.expand_exec.set_profile(true);
//             }
//
//             exec0.expand_exec.run_async();
//         }
//
//         println!("Synchronizing");
//         exec0.expand_exec.stream().synchronize();
//     }
//
//     println!("Profile:\n{}", exec0.expand_exec.last_profile().unwrap());
//
//     let throughput = (batch_size * iterations) as f32 / (start.elapsed().as_secs_f32());
//     println!("Throughput: {}", throughput);
// }

fn main() {}
