use std::collections::HashMap;

use itertools::Itertools;

use kn_cuda_eval::device_tensor::DeviceTensor;
use kn_cuda_sys::wrapper::handle::{CudaStream, Device};
use kn_cuda_sys::wrapper::mem::pinned::PinnedMem;
use kn_cuda_sys::wrapper::rtc::args::KernelArgs;
use kn_cuda_sys::wrapper::rtc::core::CuModule;
use kn_cuda_sys::wrapper::status::Status;
use kn_graph::dtype::DType;

fn main() {
    unsafe { main_inner() }
}

const KERNEL_SOURCE: &str = r#"
__global__ void kernel(float *a, int offset)
{
  int i = offset + threadIdx.x + blockIdx.x*blockDim.x;
  float x = a[i];
  for (int k = 0; k < 64; k++) {
      x = sqrtf(sinf(x));
  }
  a[i] = x;
}
"#;

unsafe fn main_inner() {
    let device = Device::new(0);
    println!("{:#?}", device.properties());
    println!("{:?}", device.compute_capability());

    let repeats = 16;

    let block_size = 256;
    let stream_count = 4;

    let part_size = 4 * 1024 * block_size;
    let full_size = part_size * stream_count;

    let stream = CudaStream::new(device);
    let streams = (0..stream_count).map(|_| CudaStream::new(device)).collect_vec();

    let result = CuModule::from_source(device, KERNEL_SOURCE, None, &["kernel"], &HashMap::new());
    println!("{}", result.log);
    let kernel = result.get_function_by_name("kernel").unwrap().unwrap();

    let full_tensor = DeviceTensor::alloc_simple(device, vec![full_size * 4], DType::F32);
    let full_pinned = PinnedMem::alloc(full_size * 4, false);
    let mut part_pinned = (0..stream_count)
        .map(|_| PinnedMem::alloc(part_size * 4, false))
        .collect_vec();

    // baseline
    let mut args = KernelArgs::new();
    args.push(full_tensor.ptr().ptr());
    args.push(0i32);
    let args = args.finish();

    let start = stream.record_event();
    for _ in 0..repeats {
        full_tensor.ptr().copy_linear_from_host(full_pinned.as_slice());
        kernel
            .launch_kernel((full_size / block_size) as u32, block_size as u32, 0, &stream, &args)
            .unwrap();
        full_tensor.ptr().copy_linear_to_host(full_pinned.as_slice());
    }
    let end = stream.record_event();
    end.synchronize();
    println!("Baseline: {}", end.time_elapsed_since(&start) / repeats as f32);

    // async 1
    let start = stream.record_event();
    for _ in 0..repeats {
        for i in 0..stream_count {
            streams[i].wait_for_event(&start);

            let offset = i * part_size;

            let mut args = KernelArgs::new();
            args.push(full_tensor.ptr().ptr());
            args.push(offset as i32);
            let args = args.finish();

            let part_tensor = full_tensor.slice(0, offset..offset + part_size);

            part_tensor
                .ptr()
                .copy_linear_from_host_async(&part_pinned[i], &streams[i]);
            kernel
                .launch_kernel(
                    (part_size / block_size) as u32,
                    block_size as u32,
                    0,
                    &streams[i],
                    &args,
                )
                .unwrap();
            part_tensor
                .ptr()
                .copy_linear_to_host_async(&mut part_pinned[i], &streams[i]);

            stream.wait_for_event(&streams[i].record_event());
        }
    }
    let end = stream.record_event();
    end.synchronize();
    println!("Async 1: {}", end.time_elapsed_since(&start) / repeats as f32);

    // async 2
    let start = stream.record_event();
    for i in 0..stream_count {
        streams[i].wait_for_event(&start);
    }

    for _ in 0..repeats {
        for i in 0..stream_count {
            // streams[i].wait_for_event(&start);

            let offset = i * part_size;
            let part_tensor = full_tensor.slice(0, offset..offset + part_size);

            part_tensor
                .ptr()
                .copy_linear_from_host_async(&part_pinned[i], &streams[i]);
        }

        for i in 0..stream_count {
            let offset = i * part_size;

            let mut args = KernelArgs::new();
            args.push(full_tensor.ptr().ptr());
            args.push(offset as i32);
            let args = args.finish();

            kernel
                .launch_kernel(
                    (part_size / block_size) as u32,
                    block_size as u32,
                    0,
                    &streams[i],
                    &args,
                )
                .unwrap();
        }

        for i in 0..stream_count {
            let offset = i * part_size;
            let part_tensor = full_tensor.slice(0, offset..offset + part_size);

            part_tensor
                .ptr()
                .copy_linear_to_host_async(&mut part_pinned[i], &streams[i]);

            // stream.wait_for_event(&streams[i].record_event());
        }
    }

    for i in 0..stream_count {
        stream.wait_for_event(&streams[i].record_event());
    }

    let end = stream.record_event();
    end.synchronize();
    println!("Async 2: {}", end.time_elapsed_since(&start) / repeats as f32);
}
