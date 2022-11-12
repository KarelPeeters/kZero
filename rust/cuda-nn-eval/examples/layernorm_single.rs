use bytemuck::cast_slice;
use cuda_nn_eval::autokernel::layernorm::LayernormKernel;
use cuda_nn_eval::device_tensor::DeviceTensor;
use cuda_sys::wrapper::handle::{CudaStream, Device};
use itertools::Itertools;
use rand::{thread_rng, Rng};

fn main() {
    // settings
    let size_inert = 16384;
    let size_op = 256;

    // let size_inert = 32;
    // let size_op = 2097152;

    let cache = true;
    let reduce_thread = false;

    let iterations = 10;

    println!("Initialization");
    let device = Device::new(0);
    let stream = CudaStream::new(device);

    let elements = size_inert * size_op;
    let input_tensor = DeviceTensor::alloc_simple(device, vec![size_inert, size_op]);
    let output_tensor = DeviceTensor::alloc_simple(device, vec![size_inert, size_op]);

    let mut rng = thread_rng();
    let random_data = (0..elements).map(|_| rng.gen::<f32>()).collect_vec();
    unsafe {
        input_tensor.copy_simple_from_host(cast_slice(&random_data));
    }

    println!("Submitting profiles");
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
        reduce_thread,
    );

    // warmup
    for _ in 0..iterations {
        unsafe {
            kernel.run(&stream, &input_tensor, None, &output_tensor);
        }
    }

    let start = stream.record_event();
    for _ in 0..iterations {
        unsafe {
            kernel.run(&stream, &input_tensor, None, &output_tensor);
        }
    }
    let end = stream.record_event();

    println!("Waiting for completion");
    stream.synchronize();

    println!("Extracting timings");
    let time = end.time_elapsed_since(&start);
    let throughput = (iterations * elements) as f32 / time / 1024f32.powi(3);

    println!(
        "  Size {} x {}: time {} throughput {} Gel/s",
        size_inert, size_op, time, throughput
    );
}
