use nn_graph::graph::Graph;
use nn_graph::shape::Shape;

use crate::root::runner::test_all;
use crate::root::tensor_utils::{manual_tensor, range_vec};

#[test]
fn single_element_affine() {
    let input_data = manual_tensor((8, 1, 1, 1), range_vec(8));
    let output_data = input_data.map(|&x| ((x + 1.0) * 2.0 * 10.0 + 3.0) * 4.0).to_shared();

    let mut graph = Graph::new();

    let const_shape = Shape::fixed(&[1, 1, 1, 1]);
    let bias_0 = graph.constant(const_shape.clone(), vec![1.0]);
    let scale_0 = graph.constant(const_shape.clone(), vec![2.0]);
    let filter = graph.constant(const_shape.clone(), vec![10.0]);
    let bias_1 = graph.constant(const_shape.clone(), vec![3.0]);
    let scale_1 = graph.constant(const_shape.clone(), vec![4.0]);

    let curr = graph.input(Shape::fixed(&input_data.shape()));
    let curr = graph.add(curr, bias_0);
    let curr = graph.mul(curr, scale_0);
    let curr = graph.conv(curr, filter, 0);
    let curr = graph.add(curr, bias_1);
    let curr = graph.mul(curr, scale_1);
    graph.output(curr);

    test_all(
        &graph,
        0,
        &[input_data],
        Some(&[output_data]),
    )
}

#[test]
fn single_element_multi_channel_affine() {
    let input_data = manual_tensor((8, 3, 1, 1), range_vec(8*3));

    let mut graph = Graph::new();

    let before_shape = Shape::fixed(&[1, 3, 1, 1]);
    let after_shape = Shape::fixed(&[1, 2, 1, 1]);
    let filter_shape = Shape::fixed(&[2, 3, 1, 1]);

    let bias_0 = graph.constant(before_shape.clone(), vec![1.0, 2.0, 3.0]);
    let scale_0 = graph.constant(before_shape.clone(), vec![2.0, 3.0, 4.0]);
    let filter = graph.constant(filter_shape.clone(), vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    let bias_1 = graph.constant(after_shape.clone(), vec![3.0, 4.0]);
    let scale_1 = graph.constant(after_shape.clone(), vec![4.0, 5.0]);

    let curr = graph.input(Shape::fixed(&input_data.shape()));
    let curr = graph.add(curr, bias_0);
    let curr = graph.mul(curr, scale_0);
    let curr = graph.conv(curr, filter, 0);
    let curr = graph.add(curr, bias_1);
    let curr = graph.mul(curr, scale_1);
    graph.output(curr);

    test_all(
        &graph,
        0,
        &[input_data],
        None,
    )
}