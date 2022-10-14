use std::ptr::null_mut;

use cuda_sys::wrapper::handle::{CudaStream, Device};
use cuda_sys::wrapper::rtc::args::KernelArgs;
use cuda_sys::wrapper::rtc::core::{CuFunction, Dim3};
use cuda_sys::wrapper::status::Status;

use crate::autokernel::common::{
    c_array_string, c_nested_array_string, compile_cached_kernel, fill_replacements, DisplayCFloat, KernelKey,
};
use crate::device_tensor::DeviceTensor;
use crate::shape::StridedShape;

#[derive(Debug)]
pub struct LayernormKernel {
    input_shape: StridedShape,
    output_shape: StridedShape,

    _norm_axis: usize,
    _static_size: usize,

    _eps: f32,
    _alpha0: f32,
    _alpha1: f32,
    _beta: f32,

    function: CuFunction,

    blocks: u32,
    threads_per_block: u32,
}

const LAYERNORM_SOURCE: &str = include_str!("layernorm.cu");

impl LayernormKernel {
    pub fn new(
        device: Device,
        input_shape: &StridedShape,
        output_shape: &StridedShape,
        norm_axis: usize,
        eps: f32,
        alpha_0: f32,
        alpha_1: f32,
        beta: f32,
        cache: bool,
        reduce_thread: bool,
    ) -> Self {
        assert_eq!(input_shape.shape(), output_shape.shape());

        let norm_size = input_shape.shape()[norm_axis];
        let static_size = input_shape.size() / norm_size;

        let input_static = input_shape.remove(norm_axis);
        let output_static = output_shape.remove(norm_axis);

        let static_dense = StridedShape::new_simple(input_static.shape().to_vec());

        let mut static_strides = [input_static.strides().to_vec(), output_static.strides().to_vec()];
        let mut static_dense_strides = static_dense.strides().to_vec();

        let norm_strides = [input_shape.strides()[norm_axis], output_shape.strides()[norm_axis]];

        // pad arrays to ensure they never become zero-sized
        static_strides[0].push(0);
        static_strides[1].push(0);
        static_dense_strides.push(1);

        let blocks = static_size as u32;
        let threads_per_block = 512;

        let replacements = vec![
            ("$THREADS_PER_BLOCK$", format!("{}", threads_per_block)),
            ("$CACHE$", format!("{}", cache)),
            ("$RANK$", format!("{}", input_shape.rank())),
            ("$STATIC_SIZE$", format!("{}", static_size)),
            ("$NORM_SIZE$", format!("{}", norm_size)),
            ("$EPS$", format!("{}", DisplayCFloat(eps))),
            ("$ALPHA_0$", format!("{}", DisplayCFloat(alpha_0))),
            ("$ALPHA_1$", format!("{}", DisplayCFloat(alpha_1))),
            ("$BETA$", format!("{}", DisplayCFloat(beta))),
            ("$STATIC_DENSE_STRIDES$", c_array_string(&static_dense_strides)),
            ("$STATIC_STRIDES$", c_nested_array_string(&static_strides)),
            ("$NORM_STRIDES$", c_array_string(&norm_strides)),
        ];

        // compile the kernel
        let source = fill_replacements(LAYERNORM_SOURCE, &replacements);
        let key = KernelKey {
            device,
            source,
            func_name: "layernorm_kernel".to_owned(),
        };
        let function = compile_cached_kernel(key);

        // wrap everything up
        LayernormKernel {
            function,
            input_shape: input_shape.clone(),
            output_shape: output_shape.clone(),
            _norm_axis: norm_axis,
            _static_size: static_size,
            _eps: eps,
            _alpha0: alpha_0,
            _alpha1: alpha_1,
            _beta: beta,

            blocks,
            threads_per_block,
        }
    }

    pub unsafe fn run(
        &self,
        stream: &CudaStream,
        input0: &DeviceTensor,
        input1: Option<&DeviceTensor>,
        output: &DeviceTensor,
    ) {
        assert_eq!(input0.strided_shape(), &self.input_shape);
        if let Some(input1) = input1 {
            assert_eq!(input1.strided_shape(), &self.input_shape);
        }
        assert_eq!(output.strided_shape(), &self.output_shape);

        if self._alpha1 != 0.0 {
            assert_eq!(input1.is_some(), true);
        }

        let mut args = KernelArgs::new();
        args.push(input0.ptr().ptr());
        args.push(input1.map_or(null_mut(), |x| x.ptr().ptr()));
        args.push(output.ptr().ptr());
        let args = args.finish();

        self.function
            .launch_kernel(
                Dim3::single(self.blocks),
                Dim3::single(self.threads_per_block),
                0,
                &stream,
                &args,
            )
            .unwrap();
    }
}
