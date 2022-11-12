use bytemuck::cast_slice;
use itertools::Itertools;
use rand::{thread_rng, Rng};

use cuda_nn_eval::autokernel::layernorm::LayernormKernel;
use cuda_nn_eval::device_tensor::DeviceTensor;
use cuda_sys::wrapper::event::CudaEvent;
use cuda_sys::wrapper::handle::{CudaStream, Device};

fn main() {
    // settings
    let sizes_inert = (4..20).map(|p| 1 << p).collect_vec();
    let sizes_op = (6..20).map(|p| 1 << p).collect_vec();
    let caches = vec![false, true];

    let max_elements = 1024 * 1024 * 1024 / 4;
    let max_cache_size = 4 * 1024;
    let iterations = 100;

    println!("Initialization");
    let device = Device::new(0);
    let stream = CudaStream::new(device);

    let max_input_buffer = DeviceTensor::alloc_simple(device, vec![max_elements]);
    let max_output_buffer = DeviceTensor::alloc_simple(device, vec![max_elements]);

    let mut rng = thread_rng();
    let random_data = (0..max_elements).map(|_| rng.gen::<f32>()).collect_vec();
    unsafe {
        max_input_buffer.copy_simple_from_host(cast_slice(&random_data));
    }

    let skip = |size_inert: usize, size_op: usize, cache: bool| {
        size_inert * size_op > max_elements || (cache && size_op > max_cache_size)
    };

    let expected_entries = itertools::iproduct!(&sizes_inert, &sizes_op, &caches)
        .filter(|(&size_inert, &size_op, &cache)| !skip(size_inert, size_op, cache))
        .count();

    println!("Submitting profiles");
    let mut entries = vec![];
    let mut completed_entries = 0;

    for (&size_inert, &size_op, &cache) in itertools::iproduct!(&sizes_inert, &sizes_op, &caches) {
        let elements = size_inert * size_op;
        if skip(size_inert, size_op, cache) {
            entries.push(None);
            continue;
        }

        completed_entries += 1;
        let progress = completed_entries as f32 / expected_entries as f32;
        println!(
            "  Compiling {}x{} {} (progress {:.2})",
            size_inert, size_op, cache, progress
        );

        let input_tensor = max_input_buffer
            .slice(0, 0..elements)
            .view(vec![size_inert, size_op])
            .unwrap();
        let output_tensor = max_output_buffer
            .slice(0, 0..elements)
            .view(vec![size_inert, size_op])
            .unwrap();

        let kernel = LayernormKernel::new(
            device,
            input_tensor.strided_shape(),
            output_tensor.strided_shape(),
            1,
            1e-5,
            1.0,
            0.0,
            1.0,
            cache,
            false,
        );

        // let kernel = ScalarKernel::new_for_shapes(
        //     device,
        //     "*x0 = *x1;",
        //     &[
        //         output_tensor.strided_shape().clone(),
        //         input_tensor.strided_shape().clone(),
        //     ],
        // );

        let start = stream.record_event();
        for _ in 0..iterations {
            unsafe {
                kernel.run(&stream, &input_tensor, None, &output_tensor);
                // kernel.run(&stream, &[output_tensor.clone(), input_tensor.clone()]);
            }
        }
        let end = stream.record_event();

        let entry = ProfileEntry {
            size_inert,
            size_op,
            cache,
            start,
            end,
        };
        entries.push(Some(entry));
    }

    if completed_entries != expected_entries {
        eprintln!(
            "Warning: entry count mismatch {} vs {}",
            entries.len(),
            expected_entries
        );
    }

    println!("Waiting for completion");
    stream.synchronize();

    println!("Extracting timings");
    let mut throughputs = vec![];
    for entry in entries {
        if let Some(entry) = entry {
            let ProfileEntry {
                size_inert,
                size_op,
                cache: _,
                start,
                end,
            } = entry;

            let elements = size_inert * size_op;
            let time = end.time_elapsed_since(&start);
            let throughput = (iterations * elements) as f32 / time / 1024f32.powi(3);

            println!(
                "  Size {} x {}: time {} throughput {} Gel/s",
                size_inert, size_op, time, throughput
            );

            throughputs.push(throughput);
        } else {
            throughputs.push(f32::NAN);
        }
    }

    println!("Final data:");

    println!("size_inert = {:?}", sizes_inert);
    println!("size_op = {:?}", sizes_op);
    println!("caches = {:?}", caches);
    println!("throughput = {:?}", throughputs);
}

#[derive(Debug)]
#[allow(dead_code)]
struct ProfileEntry {
    size_inert: usize,
    size_op: usize,
    cache: bool,
    start: CudaEvent,
    end: CudaEvent,
}
