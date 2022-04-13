use cuda_nn_eval::tensor::DeviceTensor;
use cuda_sys::wrapper::descriptor::ConvolutionDescriptor;
use cuda_sys::wrapper::group::FusedConvolutionArgs;
use cuda_sys::wrapper::handle::{CudnnHandle, Device};
use cuda_sys::wrapper::operation::{run_conv, STANDARD_CONV_ALGO};

fn main() {
    let device = Device::new(0);
    let handle = CudnnHandle::new(device);

    let b = 64;
    let s = 128;
    let c = 256;

    let iterations = 128;

    for (k, p) in [(3, 2), (2, 1)] {
        println!("Using k={}, p={}", k, p);

        let input = DeviceTensor::alloc_simple(device, vec![b, c, s, s]);
        let filter = DeviceTensor::alloc_simple(device, vec![c, c, k, k]);
        let output = DeviceTensor::alloc_simple(device, vec![b, c, s - p, s - p]);

        let input_desc = input.shape.descriptor();
        let filter_desc = filter.shape.filter_descriptor();
        let output_desc = output.shape.descriptor();

        let conv_desc = ConvolutionDescriptor::new(0, 0, 1, 1, 1, 1);

        let algo = STANDARD_CONV_ALGO;
        let work_size = conv_desc.workspace_size(&handle, algo, &input_desc, &filter_desc, &output_desc);
        let work_buffer = device.alloc(work_size);

        unsafe {
            let start = handle.stream().record_new_event();
            let mut events = vec![];

            for i in 0..iterations {
                println!("Launching {}", i);
                run_conv(
                    &handle,
                    &conv_desc,
                    algo,
                    &work_buffer,
                    work_size,
                    &filter_desc,
                    &filter.ptr,
                    &input_desc,
                    &input.ptr,
                    &output_desc,
                    &output.ptr,
                );

                events.push(handle.stream().record_new_event());
            }

            let end = handle.stream().record_new_event();

            for i in 0..iterations {
                events[i].synchronize();
                println!("Completed {}", i);
            }

            end.synchronize();
            let delta = end.time_elapsed_since(&start);

            println!("Time: {}", delta);
            println!("Throughput: {} convs/s", iterations as f32 / delta);
        }
    }
}
