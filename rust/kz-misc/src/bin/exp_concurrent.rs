// use std::ffi::c_void;
// use std::time::Instant;
//
// use kn_cuda_sys::bindings::{cudaMemcpyAsync, cudaMemcpyKind};
// use kn_cuda_sys::wrapper::handle::Device;
// use kn_cuda_sys::wrapper::mem::pinned::PinnedMem;
// use kn_cuda_sys::wrapper::status::Status;
// use kz_core::mapping::chess::ChessStdMapper;
// use kz_core::network::muzero::MuZeroGraphs;
//
// fn main() {
//     let path = r#"D:/Documents/A0/muzero/limit64_reshead/models_8000_"#;
//     let device = Device::new(0);
//     let batch_size = 256;
//     let mapper = ChessStdMapper;
//
//     println!("Loading graphs");
//     let graphs = MuZeroGraphs::load(&path, mapper).unwrap();
//     let graphs = graphs.optimize(Default::default());
//     let fused = graphs.fuse(Default::default());
//     let _state_saved_size = fused.info.state_saved_shape(mapper).size().eval(1);
//
//     println!("Building executors");
//     let mut exec0 = fused.expand_executor(device, batch_size);
//     let mut exec1 = fused.expand_executor(device, batch_size);
//
//     let iterations = 20;
//     let start = Instant::now();
//
//     let x = PinnedMem::alloc(4 * 8, false);
//     let x_device = device.alloc(4 * 8);
//
//     unsafe {
//         println!("Queuing operations");
//         for i in 0..iterations {
//             println!("i={}", i);
//             exec0.expand_exec.run_async();
//             exec1.expand_exec.run_async();
//
//             cudaMemcpyAsync(
//                 x_device.ptr(),
//                 x.ptr() as *const c_void,
//                 4 * 8,
//                 cudaMemcpyKind::cudaMemcpyHostToDevice,
//                 exec0.expand_exec.stream().inner(),
//             )
//             .unwrap();
//
//             // x_device.copy_linear_from_host(cast_slice(&x));
//         }
//
//         println!("Synchronizing");
//         exec0.expand_exec.stream().synchronize();
//         exec1.expand_exec.stream().synchronize();
//     }
//
//     let throughput = (2 * batch_size * iterations) as f32 / (start.elapsed().as_secs_f32());
//     println!("Throughput: {}", throughput);
// }

fn main() {}
