use cuda_nn_eval::autokernel::common::{compile_cached_kernel, KernelKey};
use cuda_nn_eval::device_tensor::DeviceTensor;
use cuda_sys::wrapper::handle::{CudaStream, Device};
use cuda_sys::wrapper::rtc::args::KernelArgs;
use cuda_sys::wrapper::status::Status;

const SOURCE: &str = r#"

__global__ void foo(float *data) {
    
    auto data_vec = reinterpret_cast<float4*>(data+1);

    data_vec[1] = data_vec[0];
}

"#;

fn main() {
    let device = Device::new(0);
    let stream = CudaStream::new(device);

    let key = KernelKey {
        device,
        source: SOURCE.to_owned(),
        func_name: "foo".to_owned(),
    };
    let kernel = compile_cached_kernel(key);

    let element_count = 16;

    let data = DeviceTensor::alloc_simple(device, vec![element_count]);
    let mut data_buffer = vec![0.0; element_count];
    for i in 0..element_count {
        data_buffer[i] = (i + 1) as f32;
    }

    unsafe {
        data.copy_simple_from_host(&data_buffer);

        let mut args = KernelArgs::new();
        args.push(data.ptr().ptr());
        let args = args.finish();

        kernel.launch_kernel(1, 1, 0, &stream, &args).unwrap();
        stream.synchronize();

        data.copy_simple_to_host(&mut data_buffer);
    }

    println!("{:?}", data_buffer);
}
