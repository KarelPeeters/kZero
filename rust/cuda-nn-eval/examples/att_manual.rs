use std::ops::Add;

use itertools::Itertools;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rand_distr::Distribution;
use rand_distr::StandardNormal;

use cuda_nn_eval::autokernel::attention::AttentionKernel;
use cuda_nn_eval::device_tensor::DeviceTensor;
use cuda_nn_eval::tester::assert_tensors_match;
use cuda_sys::wrapper::handle::{CudaStream, Device};
use nn_graph::cpu::{cpu_eval_graph, Tensor};
use nn_graph::graph::{BinaryOp, Graph};
use nn_graph::ndarray::{s, ArcArray, Array1, Array2, Array3, Axis, Dimension, IntoDimension};
use nn_graph::optimizer::optimize_graph;
use nn_graph::shape;
use nn_graph::shape::Size;

fn main() {
    let head_dim = 8;
    let seq_len = 32;
    let batch_size = 1; // technically batch*heads

    let mut rng = SmallRng::seed_from_u64(0);
    let data_q = rng_tensor((seq_len, batch_size, head_dim), &mut rng);
    let data_k = rng_tensor((seq_len, batch_size, head_dim), &mut rng) / (head_dim as f32).sqrt();
    let data_v = rng_tensor((seq_len, batch_size, head_dim), &mut rng);

    // let graph = load_graph_from_onnx_path("ignored/att.onnx", false);
    let graph = build_att_graph(head_dim, seq_len, false);
    let graph = optimize_graph(&graph, Default::default());
    println!("{}", graph);

    // let device = Device::new(0);
    // let exec = CudaExecutor::new(device, &graph, batch_size);
    // println!("{:?}", exec);

    let output_manual = flash_att_impl(&data_q, &data_k, &data_v, seq_len, batch_size, head_dim);
    let cpu_output = cpu_eval_graph(&graph, batch_size, &[data_q, data_k, data_v]).remove(0);

    assert_tensors_match(&[cpu_output], &[output_manual], true);
}

type Array1f = Array1<f32>;
type Array2f = Array2<f32>;

fn flash_att_impl(
    global_q: &Tensor,
    global_k: &Tensor,
    global_v: &Tensor,
    seq_len: usize,
    batch_size: usize,
    head_dim: usize,
) -> Tensor {
    assert_eq!(batch_size, 1);
    assert_eq!(global_q.shape(), &[seq_len, batch_size, head_dim]);
    assert_eq!(global_k.shape(), &[seq_len, batch_size, head_dim]);
    assert_eq!(global_v.shape(), &[seq_len, batch_size, head_dim]);

    // zero/inf init is important
    let mut global_o = Array3::<f32>::zeros((seq_len, 1, head_dim));
    let mut global_max = Array1::<f32>::zeros(seq_len);
    let mut global_sum = Array1::<f32>::zeros(seq_len);
    global_max.fill(f32::NEG_INFINITY);

    let block_size_qo = 8;
    let block_size_kv = 16;
    let block_count_qo = ceil_div(seq_len, block_size_qo);
    let block_count_kv = ceil_div(seq_len, block_size_kv);

    let global_k_shaped = global_k.reshape((block_count_kv, block_size_kv, head_dim));
    let global_q_shaped = global_q.reshape((block_count_qo, block_size_qo, head_dim));
    let global_v_shaped = global_v.reshape((block_count_kv, block_size_kv, head_dim));
    let mut global_o_shaped = global_o
        .view_mut()
        .into_shape((block_count_qo, block_size_qo, head_dim))
        .unwrap();
    let mut global_max_shaped = global_max
        .view_mut()
        .into_shape((block_count_qo, block_size_qo))
        .unwrap();
    let mut global_sum_shaped = global_sum
        .view_mut()
        .into_shape((block_count_qo, block_size_qo))
        .unwrap();

    // TODO relax this later
    assert!(seq_len % block_size_qo == 0 && seq_len % block_size_kv == 0);

    for block_kv_j in 0..block_count_kv {
        // load kj, vj

        let kj = global_k_shaped.slice(s![block_kv_j, .., ..]).to_owned();
        let vj = global_v_shaped.slice(s![block_kv_j, .., ..]).to_owned();

        for block_qo_i in 0..block_count_qo {
            // load inputs from global memory
            let qi = global_q_shaped.slice(s![block_qo_i, .., ..]).to_owned();
            let o_old = global_o_shaped.slice(s![block_qo_i, .., ..]).to_owned();
            let sum_old = global_sum_shaped.slice(s![block_qo_i, ..]).to_owned();
            let max_old = global_max_shaped.slice(s![block_qo_i, ..]).to_owned();

            // compute all deltas and new values
            let logit_delta: Array2f = qi.dot(&kj.view().permuted_axes([1, 0]));
            let max_delta: Array1f = logit_delta.fold_axis(Axis(1), f32::NEG_INFINITY, fn_ref(f32::max));

            let p_delta = Array2f::from_shape_fn((block_size_qo, block_size_kv), |(a, b)| {
                (logit_delta[(a, b)] - max_delta[a]).exp()
            });
            let sum_delta: Array1f = p_delta.fold_axis(Axis(1), 0.0, fn_ref(f32::add));

            let max_new: Array1f = Array1f::from_shape_fn(block_size_qo, |a| f32::max(max_old[a], max_delta[a]));
            let sum_new = Array1::from_shape_fn(block_size_qo, |a| {
                (max_old[a] - max_new[a]).exp() * sum_old[a] + (max_delta[a] - max_new[a]).exp() * sum_delta[a]
            });

            let o_delta = p_delta.dot(&vj);
            let o_new = Array2f::from_shape_fn((block_size_qo, head_dim), |(a, d)| {
                let max_old = max_old[a];
                let max_delta = max_delta[a];
                let max_new = max_new[a];
                let o_old = o_old[(a, d)];
                let o_delta = o_delta[(a, d)];
                let sum_old = sum_old[a];
                let sum_new = sum_new[a];

                let o_old_scaled = (max_old - max_new).exp() * o_old;
                let o_new_scaled = (max_delta - max_new).exp() * o_delta;
                1.0 / sum_new * (sum_old * o_old_scaled + o_new_scaled)
            });

            // store outputs to global memory
            global_o_shaped.slice_mut(s![block_qo_i, .., ..]).assign(&o_new);

            global_sum_shaped.slice_mut(s![block_qo_i, ..]).assign(&sum_new);
            global_max_shaped.slice_mut(s![block_qo_i, ..]).assign(&max_new);
        }
    }

    global_o.into_dyn().into_shared()
}

fn fn_ref<T: Copy>(f: impl Fn(T, T) -> T) -> impl Fn(&T, &T) -> T {
    move |&a, &b| f(a, b)
}

fn _derp_cuda(seq_len: usize, batch_size: usize, head_dim: usize) {
    let device = Device::new(0);
    let stream = CudaStream::new(device);
    // let exec = CudaExecutor::new(device, &graph, batch_size);
    // println!("{:?}", exec);

    let shape = vec![seq_len, batch_size, head_dim];

    let input_q = DeviceTensor::alloc_simple(device, shape.clone());
    let input_k = DeviceTensor::alloc_simple(device, shape.clone());
    let input_v = DeviceTensor::alloc_simple(device, shape.clone());
    let output = DeviceTensor::alloc_simple(device, shape);

    let kernel = AttentionKernel::new(
        device,
        input_q.strided_shape(),
        input_k.strided_shape(),
        input_v.strided_shape(),
        output.strided_shape(),
    );

    let scratch = DeviceTensor::alloc_simple(device, vec![kernel.scratch_size()]);

    unsafe {
        kernel.run(&stream, &input_q, &input_k, &input_v, &output, &scratch);
    }
    stream.synchronize();

    println!("{:?}", kernel);
}

fn build_att_graph(d: usize, s: usize, scaled: bool) -> Graph {
    let mut graph = Graph::new();

    let q = graph.input(shape![s, Size::BATCH, d]);
    let k = graph.input(shape![s, Size::BATCH, d]);
    let v = graph.input(shape![s, Size::BATCH, d]);

    let q_perm = graph.permute(q, vec![1, 0, 2]);
    let k_perm = graph.permute(k, vec![1, 2, 0]);

    let logits_raw = graph.batched_mat_mul(q_perm, k_perm);

    let logits = if scaled {
        let scale = graph.scalar(1.0 / (d as f32).sqrt());
        graph.binary(BinaryOp::Mul, logits_raw, scale)
    } else {
        logits_raw
    };

    let weights = graph.softmax(logits, 2);

    let v_perm = graph.permute(v, vec![1, 0, 2]);
    let att_raw = graph.batched_mat_mul(weights, v_perm);
    let att = graph.permute(att_raw, vec![1, 0, 2]);

    graph.output(att);
    graph
}

pub fn rng_tensor<I: IntoDimension + Copy>(shape: I, rng: &mut impl Rng) -> Tensor {
    let size = shape.into_dimension().size();
    let data = rng_vec(size, rng);
    manual_tensor(shape, data)
}

pub fn rng_vec(len: usize, rng: &mut impl Rng) -> Vec<f32> {
    (0..len).map(|_| StandardNormal.sample(rng)).collect_vec()
}

pub fn manual_tensor<I: IntoDimension>(shape: I, data: Vec<f32>) -> Tensor {
    ArcArray::from_shape_vec(shape, data)
        .expect("Shape and data length mismatch")
        .into_dyn()
}

pub fn ceil_div(x: usize, y: usize) -> usize {
    (x + y - 1) / y
}
