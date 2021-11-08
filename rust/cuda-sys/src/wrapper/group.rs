use crate::bindings::cudnnConvolutionFwdAlgo_t;
use crate::wrapper::descriptor::{ActivationDescriptor, ConvolutionDescriptor, FilterDescriptor, TensorDescriptor, TensorOpDescriptor};
use crate::wrapper::handle::CudnnHandle;
use crate::wrapper::mem::device::DeviceMem;
use crate::wrapper::operation::{run_conv_bias_res_activation, run_tensor_op};

/// The arguments necessary for a fused convolution call.
#[derive(Debug)]
pub struct FusedConvolutionArgs {
    pub conv_desc: ConvolutionDescriptor,
    pub algo: cudnnConvolutionFwdAlgo_t,
    pub work_mem: DeviceMem,

    pub filter_desc: FilterDescriptor,
    pub filter_mem: DeviceMem,
    pub input_desc: TensorDescriptor,
    pub input_mem: DeviceMem,

    pub res_mem: Option<DeviceMem>,
    pub bias_desc: TensorDescriptor,
    pub bias_mem: DeviceMem,

    pub act_desc: ActivationDescriptor,

    pub output_desc: TensorDescriptor,
    pub output_mem: DeviceMem,
}

impl FusedConvolutionArgs {
    pub unsafe fn run(&self, handle: &mut CudnnHandle) {
        run_conv_bias_res_activation(
            handle,
            &self.act_desc,
            &self.conv_desc,
            self.algo,
            &self.work_mem,
            &self.filter_desc,
            &self.filter_mem,
            &self.input_desc,
            &self.input_mem,
            self.res_mem.as_ref(),
            &self.bias_desc,
            &self.bias_mem,
            &self.output_desc,
            &self.output_mem,
        )
    }
}

#[derive(Debug)]
pub struct TensorOpArgs {
    pub op_desc: TensorOpDescriptor,
    pub alpha_1: f32,
    pub input_1_desc: TensorDescriptor,
    pub input_1_mem: DeviceMem,
    pub alpha_2: f32,
    pub input_2_desc: TensorDescriptor,
    pub input_2_mem: DeviceMem,
    pub beta: f32,
    pub output_desc: TensorDescriptor,
    pub output_mem: DeviceMem,
}

impl TensorOpArgs {
    pub unsafe fn run(&self, handle: &mut CudnnHandle) {
        run_tensor_op(
            handle,
            &self.op_desc,
            self.alpha_1,
            &self.input_1_desc,
            &self.input_1_mem,
            self.alpha_2,
            &self.input_2_desc,
            &self.input_2_mem,
            self.beta,
            &self.output_desc,
            &self.output_mem,
        )
    }
}
