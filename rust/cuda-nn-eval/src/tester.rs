use bytemuck::{cast_slice, cast_slice_mut};
use itertools::{Itertools, zip_eq};

use cuda_sys::wrapper::handle::Device;
use nn_graph::cpu::{Tensor, TensorF};
use nn_graph::graph::{ConstantData, Graph, Type, Value};
use nn_graph::ndarray::{Dimension, IxDyn};

use crate::executor::CudnnExecutor;

/// Check that the given graph produces the correct outputs as described by `check_data`,
/// which typically comes from a `.bin` file next to the `.onnx` file.
pub fn check_cudnn(graph: &Graph, check_data_bytes: &[u8]) {
    let (batch_size, inputs, expected_outputs) = load_check_data(graph, check_data_bytes);
    let outputs = eval_cudnn(&graph, batch_size, &inputs);
    assert_outputs_match(&expected_outputs, &outputs, false);
}

const ERROR_TOLERANCE: f32 = 0.0001;

pub fn assert_outputs_match(expected: &[Tensor], actual: &[Tensor], print: bool) {
    assert_eq!(expected.len(), actual.len(), "Wrong number of outputs");

    for (i, (expected, actual)) in zip_eq(expected, actual).enumerate() {
        // shape
        assert_eq!(expected.shape(), actual.shape(), "Wrong output shape for output {}", i);

        // types
        match (expected, actual) {
            (Tensor::F32(expected), Tensor::F32(actual)) => (),
            (Tensor::I32(expected), Tensor::I32(actual)) => (),
            _ => panic!("Output type mismatch, expected {:?} got {:?}", expected, actual),
        }

        // values
        let expected_f = expected.to_f32();
        let actual_f = expected.to_f32();
        assert_tensor_equal(i, expected_f, actual_f, print);
    }
}

fn assert_tensor_equal(i: usize, expected: TensorF, actual: TensorF, print: bool) {
    let mut max_error = 0.0;

    for ((indices, &expected), &actual) in zip_eq(expected.indexed_iter(), actual.iter()) {
        let error = (expected - actual).abs();
        max_error = f32::max(max_error, error);
        assert!(
            error < ERROR_TOLERANCE,
            "Wrong output value {}, expected {} at indices {:?} in output {}",
            actual, expected, indices.slice(), i,
        );

        if print {
            println!("Output {} matched, max error {}", i, max_error);
        }
    }
}

pub fn eval_cudnn(graph: &Graph, batch_size: usize, inputs: &[Tensor]) -> Vec<Tensor> {
    let inputs = inputs.iter()
        .map(|x| x.as_byte_slice().expect("Only sliceable inputs supported in test framework"))
        .collect_vec();

    let mut executor = CudnnExecutor::new(Device::new(0), graph, batch_size);
    let gpu_outputs = executor.evaluate(&inputs);

    // turn into Tensors, using the cpu shapes
    let outputs = zip_eq(graph.outputs(), gpu_outputs)
        .map(|(&value, output)| {
            let shape = graph[value].shape.eval(batch_size);

            let data = match graph[value].ty {
                Type::F32 => ConstantData::F32(cast_slice(output).to_vec()),
                Type::I32 => ConstantData::I32(cast_slice(output).to_vec()),
            };

            Tensor::from_data(IxDyn(shape.dims.as_slice()), data)
        })
        .collect_vec();

    outputs
}

/// Load the check data into `(batch_size, inputs, expected_outputs)`.
pub fn load_check_data(graph: &Graph, check_data_bytes: &[u8]) -> (usize, Vec<Tensor>, Vec<Tensor>) {
    assert!(check_data_bytes.len() >= 1, "Check data must have at least one byte, the batch size");
    let batch_size = check_data_bytes[0] as usize;

    let mut buf = &*check_data_bytes;
    let inputs = load_check_values(graph, batch_size, &mut buf, graph.inputs());
    let expected_outputs = load_check_values(graph, batch_size, &mut buf, graph.outputs());

    assert!(buf.is_empty(), "Leftover elements in check data buffer: {}", buf.len());

    (batch_size, inputs, expected_outputs)
}

/// Load the given values from the buffer while advancing it.
fn load_check_values(graph: &Graph, batch_size: usize, buf: &mut &[u8], values: &[Value]) -> Vec<Tensor> {
    values.iter()
        .map(|&value| {
            let shape = graph[value].shape.eval(batch_size);
            let ty = graph[value].ty;
            let len_used = shape.size() * ty.size_bytes();
            let data = ConstantData::from_bytes(ty, &buf[0..len_used]);
            *buf = &buf[len_used..];

            Tensor::from_data(IxDyn(&shape.dims), data)
        })
        .collect_vec()
}
