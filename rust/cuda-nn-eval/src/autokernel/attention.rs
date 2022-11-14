use itertools::Itertools;

use cuda_sys::wrapper::handle::{CudaStream, Device};
use cuda_sys::wrapper::rtc::args::KernelArgs;
use cuda_sys::wrapper::rtc::core::CuFunction;
use cuda_sys::wrapper::status::Status;

use crate::autokernel::common::{compile_cached_kernel, fill_replacements, KernelKey};
use crate::device_tensor::DeviceTensor;
use crate::shape::StridedShape;

const ATTENTION_SOURCE: &str = include_str!("attention.cu");

#[derive(Debug)]
pub struct AttentionKernel {
    q_shape: StridedShape,
    k_shape: StridedShape,
    v_shape: StridedShape,
    o_shape: StridedShape,

    function: CuFunction,

    grid_dim: u32,
    block_dim: u32,

    scratch_size: usize,
}

impl AttentionKernel {
    pub fn new(
        device: Device,
        q_shape: &StridedShape,
        k_shape: &StridedShape,
        v_shape: &StridedShape,
        o_shape: &StridedShape,
        block_size_qo: usize,
        block_size_kv: usize,
    ) -> Self {
        assert!(q_shape.has_simple_strides());
        assert!(k_shape.has_simple_strides());
        assert!(v_shape.has_simple_strides());
        assert!(o_shape.has_simple_strides());
        let shapes = check_shapes(q_shape, k_shape, v_shape, o_shape);
        assert_eq!(shapes.d_qk, shapes.d_v);
        assert_eq!(shapes.b, 1);
        assert_eq!(shapes.s_q, shapes.s_kv);

        let d = shapes.d_qk;
        let s = shapes.s_q;

        // TODO properly compute block sizes, taking memory limits into account
        // TODO add batch size to scratch (and everything else ofc)
        let scratch_size = 2 * s;

        let replacements = vec![
            ("$S$", format!("{}", s)),
            ("$D$", format!("{}", d)),
            ("$B_QO$", format!("{}", block_size_qo)),
            ("$B_KV$", format!("{}", block_size_kv)),
            ("$SCRATCH_SIZE$", format!("{}", scratch_size)),
        ];

        let source = fill_replacements(ATTENTION_SOURCE, &replacements);

        let key = KernelKey {
            device,
            source,
            func_name: "attention_kernel".to_owned(),
        };
        let function = compile_cached_kernel(key);

        AttentionKernel {
            q_shape: q_shape.clone(),
            k_shape: k_shape.clone(),
            v_shape: v_shape.clone(),
            o_shape: o_shape.clone(),
            grid_dim: 1,
            block_dim: (block_size_qo * block_size_kv) as u32,
            scratch_size,
            function,
        }
    }

    pub fn scratch_size(&self) -> usize {
        self.scratch_size
    }

    pub unsafe fn run(
        &self,
        stream: &CudaStream,
        q: &DeviceTensor,
        k: &DeviceTensor,
        v: &DeviceTensor,
        o: &DeviceTensor,
        scratch: &DeviceTensor,
    ) {
        assert_eq!(q.strided_shape(), &self.q_shape);
        assert_eq!(k.strided_shape(), &self.k_shape);
        assert_eq!(v.strided_shape(), &self.v_shape);
        assert_eq!(o.strided_shape(), &self.o_shape);
        assert_eq!(
            scratch.strided_shape(),
            &StridedShape::new_simple(vec![self.scratch_size])
        );

        let mut args = KernelArgs::new();
        args.push(q.ptr().ptr());
        args.push(k.ptr().ptr());
        args.push(v.ptr().ptr());
        args.push(o.ptr().ptr());
        args.push(scratch.ptr().ptr());
        let args = args.finish();

        self.function
            .launch_kernel(self.grid_dim, self.block_dim, 0, stream, &args)
            .unwrap();
    }
}

#[derive(Debug, Copy, Clone)]
struct AttShapes {
    b: usize,
    s_q: usize,
    s_kv: usize,
    d_qk: usize,
    d_v: usize,
}

fn check_shapes(
    q_shape: &StridedShape,
    k_shape: &StridedShape,
    v_shape: &StridedShape,
    o_shape: &StridedShape,
) -> AttShapes {
    let (s_q_0, b_0, d_qk_0) = unwrap_3(q_shape.shape());
    let (s_kv_0, b_1, d_qk_1) = unwrap_3(k_shape.shape());
    let (s_kv_1, b_2, d_v_0) = unwrap_3(v_shape.shape());
    let (s_q_1, b_3, d_v_1) = unwrap_3(o_shape.shape());

    assert!(b_0 == b_1 && b_1 == b_2 && b_2 == b_3);
    assert!(s_q_0 == s_q_1 && s_kv_0 == s_kv_1);
    assert!(d_qk_0 == d_qk_1 && d_v_0 == d_v_1);

    AttShapes {
        b: b_0,
        s_q: s_q_0,
        s_kv: s_kv_0,
        d_qk: d_qk_0,
        d_v: d_v_0,
    }
}

fn unwrap_3(shape: &[usize]) -> (usize, usize, usize) {
    assert_eq!(shape.len(), 3);
    shape.iter().copied().collect_tuple().unwrap()
}