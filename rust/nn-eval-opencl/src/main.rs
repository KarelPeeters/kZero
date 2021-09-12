use std::fmt::Write;
use std::time::Instant;

use opencl3::command_queue::{CL_QUEUE_PROFILING_ENABLE, CommandQueue};
use opencl3::context::Context;
use opencl3::device::{CL_DEVICE_TYPE_GPU, Device};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{CL_MAP_READ, CL_MAP_WRITE};
use opencl3::program::{CL_STD_3_0, Program};
use opencl3::svm::SvmVec;
use opencl3::types::{CL_BLOCKING, cl_int, CL_NON_BLOCKING};

fn main() {
    const PROGRAM_SOURCE: &str = include_str!("kernels/conv.cl");

    //pick a platform
    let platforms = opencl3::platform::get_platforms().unwrap();
    for p in &platforms {
        println!("Found platform {}", p.name().unwrap());
    }
    let platform = platforms.iter()
        .find(|p| p.name().unwrap().contains("CUDA"))
        .expect("Couldn't find matching platform");

    //pick a device
    let device = *platform.get_devices(CL_DEVICE_TYPE_GPU).unwrap()
        .first().expect("No device found in platform");
    let device = Device::new(device);
    println!("Using device {:?}", device.name().unwrap());

    println!("Max local mem size: {:?}", device.local_mem_size().unwrap());
    println!("  max work group size: {:?}", device.max_work_group_size().unwrap());
    println!("  preferred group size multiple: {:?}", device.preferred_work_group_size_multiple().unwrap());

    //print some info
    println!("  version: {}", device.version().unwrap());
    println!("  c_version: {}", device.opencl_c_version().unwrap());
    println!("  svm_cap: {}", device.svm_mem_capability());

    const KERNEL_NAME: &str = "convolution";

    // Create a Context on an OpenCL device
    let context = Context::from_device(&device).expect("Context::from_device failed");

    let batch_size = 128;
    let conv_count = 20;
    let image_width = 7;
    let filter_size = 3;
    let half_filter_size = filter_size / 2;
    let channels = 32;
    let res = false;
    let relu = false;

    // let local_size = batch_size * channels * filter_size * filter_size;
    // println!("local size: {}", local_size);

    // Build the OpenCL program source and create the kernel.
    let mut options = String::new();
    options += CL_STD_3_0;
    if res { options += "-D RES "; }
    if relu { options += "-D RELU "; }
    write!(
        &mut options,
        "-D F={} -D HF={} -D S={} -D C={}",
        filter_size, half_filter_size, image_width, channels
    ).unwrap();

    let mut program = Program::create_from_source(&context, PROGRAM_SOURCE)
        .unwrap_or_else(|e| {
            eprintln!("Failed to compile kernel:\n{}", e);
            panic!();
        });
    let build_result = program.build(context.devices(), &options);
    match build_result {
        Ok(()) => {}
        Err(e) => {
            eprintln!("Program build error: {} {:?}", e, e);
            eprintln!("{}", program.get_build_log(context.devices()[0]).unwrap());
            panic!();
        }
    }

    let kernel = Kernel::create(&program, KERNEL_NAME).expect("Kernel::create failed");

    // Create a command_queue on the Context's device
    let queue = CommandQueue::create_with_properties(
        &context,
        context.default_device(),
        CL_QUEUE_PROFILING_ENABLE,
        0,
    )
        .expect("CommandQueue::create_with_properties failed");

    // The input data
    let image_len = (batch_size * channels * image_width * image_width) as usize;
    let filter_len = (channels * channels * filter_size * filter_size) as usize;

    let mut main_svm = SvmVec::<f32>::allocate(&context, image_len)
        .expect("SVM allocation failed");
    let mut second_svm = SvmVec::<f32>::allocate(&context, image_len)
        .expect("SVM allocation failed");

    let mut filter_svm = SvmVec::<f32>::allocate(&context, filter_len)
        .expect("SVM allocation failed");

    let needs_map = !main_svm.is_fine_grained();

    if needs_map {
        queue.enqueue_svm_map(CL_BLOCKING, CL_MAP_WRITE, &mut filter_svm, &[]).unwrap();
    }
    let filter_array = vec![1.0 / (filter_size * filter_size * channels) as f32; filter_len];
    filter_svm.clone_from_slice(&filter_array);
    if needs_map {
        queue.enqueue_svm_unmap(&filter_svm, &[]).unwrap();
    }

    loop {
        let start = Instant::now();

        // Map input if not a CL_MEM_SVM_FINE_GRAIN_BUFFER
        if needs_map {
            queue.enqueue_svm_map(CL_BLOCKING, CL_MAP_WRITE, &mut main_svm, &[]).unwrap();
        }

        // Copy input data into the OpenCL SVM vector
        let input_array = vec![1.0; image_len];
        main_svm.clone_from_slice(&input_array);

        // Unmap test_values if not a CL_MEM_SVM_FINE_GRAIN_BUFFER
        if needs_map {
            queue.enqueue_svm_unmap(&main_svm, &[]).unwrap();
        }

        // Run the kernel on the input data
        for _ in 0..conv_count {
            ExecuteKernel::new(&kernel)
                .set_arg(&(batch_size as cl_int))
                .set_arg_svm(main_svm.as_mut_ptr())
                .set_arg_svm(filter_svm.as_mut_ptr())
                .set_arg_svm(second_svm.as_mut_ptr())
                .set_global_work_sizes(&[channels as usize, image_width as usize, image_width as usize])
                .set_local_work_sizes(&[channels, 1, 1])
                .enqueue_nd_range(&queue).unwrap();

            std::mem::swap(&mut main_svm, &mut second_svm);
        }

        // Wait for the kernel to complete execution on the device
        // kernel_event.wait().unwrap();

        // Map results if not a CL_MEM_SVM_FINE_GRAIN_BUFFER
        if needs_map {
            queue.enqueue_svm_map(CL_NON_BLOCKING, CL_MAP_READ, &mut main_svm, &[]).unwrap();
        }

        // wait for everything to complete
        let marker = queue.enqueue_marker_with_wait_list(&[]).unwrap();
        marker.wait().unwrap();

        // Can access OpenCL SVM directly, no need to map or read the results
        // println!("sum results: {:?}", main_svm);

        // Unmap results if not a CL_MEM_SVM_FINE_GRAIN_BUFFER
        if needs_map {
            let unmap_results_event = queue.enqueue_svm_unmap(&main_svm, &[]).unwrap();
            unmap_results_event.wait().unwrap();
        }

        let end = Instant::now();
        let delta = (end - start).as_secs_f32();
        let throughput = batch_size as f32 / delta;
        println!("Took {}s, throughput: {:.2}", delta, throughput);
    }
}