use opencl3::command_queue::{CL_QUEUE_PROFILING_ENABLE, CommandQueue};
use opencl3::context::Context;
use opencl3::device::{CL_DEVICE_TYPE_GPU, Device};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::program::{CL_STD_2_0, Program};
use opencl3::svm::SvmVec;
use opencl3::types::cl_int;

fn main() -> opencl3::Result<()> {
    const PROGRAM_SOURCE: &str = r#"
        kernel void inclusive_scan_int (global int* output,
                                        global int const* values)
        {
            int sum = 0;
            size_t lid = get_local_id(0);
            size_t lsize = get_local_size(0);

            size_t num_groups = get_num_groups(0);
            for (size_t i = 0u; i < num_groups; ++i)
            {
                size_t lidx = i * lsize + lid;
                int value = work_group_scan_inclusive_add(values[lidx]);
                output[lidx] = sum + value;

                sum += work_group_broadcast(value, lsize - 1);
            }
        }
    "#;

    //pick a platform
    let platforms = opencl3::platform::get_platforms()?;
    for p in &platforms {
        println!("Found platform {}", p.name()?);
    }
    let platform = platforms.iter()
        .find(|p| p.name().unwrap().contains("CUDA"))
        .expect("Couldn't find matching platform");

    //pick a device
    let device = *platform.get_devices(CL_DEVICE_TYPE_GPU)?
        .first().expect("No device found in platform");
    let device = Device::new(device);
    println!("Using device {:?}", device.name()?);

    //print some info
    println!("  version: {}", device.version()?);
    println!("  c_version: {}", device.opencl_c_version()?);
    println!("  svm_cap: {}", device.svm_mem_capability());

    let svm_capability = device.svm_capabilities()?;

    const KERNEL_NAME: &str = "inclusive_scan_int";

    // Create a Context on an OpenCL device
    let context = Context::from_device(&device).expect("Context::from_device failed");

    // Build the OpenCL program source and create the kernel.
    let program = Program::create_and_build_from_source(&context, PROGRAM_SOURCE, CL_STD_2_0)
        .expect("Program::create_and_build_from_source failed");
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
    const ARRAY_SIZE: usize = 8;
    let value_array: [cl_int; ARRAY_SIZE] = [3, 2, 5, 9, 7, 1, 4, 2];

    // Copy input data into an OpenCL SVM vector
    let mut test_values = SvmVec::<cl_int>::with_capacity(&context, svm_capability, ARRAY_SIZE);
    for &val in value_array.iter() {
        test_values.push(val);
    }

    // The output data, an OpenCL SVM vector
    let mut results =
        SvmVec::<cl_int>::with_capacity_zeroed(&context, svm_capability, ARRAY_SIZE);
    unsafe { results.set_len(ARRAY_SIZE) };

    // Run the kernel on the input data
    let kernel_event = ExecuteKernel::new(&kernel)
        .set_arg_svm(results.as_mut_ptr())
        .set_arg_svm(test_values.as_ptr())
        .set_global_work_size(ARRAY_SIZE)
        .enqueue_nd_range(&queue)
        .unwrap();

    // Wait for the kernel to complete execution on the device
    kernel_event.wait().unwrap();

    // Can access OpenCL SVM directly, no need to map or read the results
    println!("sum results: {:?}", results);

    Ok(())
}