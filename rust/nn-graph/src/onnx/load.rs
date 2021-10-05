use std::path::Path;

use byteorder::{ByteOrder, LittleEndian};
use itertools::Itertools;
use prost::Message;

use crate::graph::{Graph, Value};
use crate::onnx::attributes::Attributes;
use crate::onnx::proto::{ModelProto, tensor_shape_proto, TensorProto, TypeProto};
use crate::onnx::proto::tensor_proto::DataType;
use crate::onnx::proto::tensor_shape_proto::dimension::Value as ProtoDimValue;
use crate::onnx::proto::type_proto::Value as ProtoTypeValue;
use crate::onnx::store::Store;
use crate::shape::{Shape, Size};

pub fn load_onnx_impl(path: &Path) -> Graph {
    let model = load_model_proto(path);
    let model_graph = model.graph.unwrap();

    for init in &model_graph.initializer {
        assert_eq!(DataType::Float as i32, init.data_type);
    }

    let mut graph = Graph::new();

    let mut nodes = Store::default();

    load_initializers(&mut graph, &mut nodes, &model_graph.initializer);

    for input in &model_graph.input {
        // initializers are allowed to re-appear in the inputs, so we skip them the second time
        if nodes.contains(&&*input.name) {
            continue;
        }

        let shape = resolve_tensor_shape(input.r#type.as_ref().unwrap());
        let value = graph.input(shape);
        nodes.define(&input.name, value);
    }

    for node in &model_graph.node {
        assert_eq!(1, node.output.len(), "nodes with multiple outputs not supported");
        let output_name = &node.output[0];

        let mut attrs = Attributes::from(&node.attribute);
        let inputs: Vec<Value> = node.input.iter().enumerate().map(|(i, name)| {
            *nodes.get(name)
                .unwrap_or_else(|| panic!("Input {} {} of node {} not found", i, name, node.name))
        }).collect_vec();

        let value = match &*node.op_type {
            "Conv" => {
                assert!(inputs.len() <= 3);
                let input = inputs[0];
                let filter = inputs[1];
                let bias = inputs.get(2).copied();

                let g = attrs.take_int("group");
                let [kw, kh] = unwrap_2(attrs.take_ints("kernel_shape"));
                let [ph0, pv0, ph1, pv1] = unwrap_4(attrs.take_ints("pads"));
                let [sw, sh] = unwrap_2(attrs.take_ints("strides"));
                let [dw, dh] = unwrap_2(attrs.take_ints("dilations"));

                let [_, _, kernel_w, kernel_h] =
                    graph[filter].shape.unwrap_fixed().unwrap_4();

                assert_eq!(1, g);
                assert!(ph0 == ph1 && pv0 == pv1 && ph0 == pv0);
                assert!(dw == 1 && dh == 1);
                assert!(sw == 1 && sh == 1);
                assert!(kernel_w == kw && kernel_h == kh);

                let conv = graph.conv(input, filter, ph0);

                if let Some(bias) = bias {
                    let bias_size = graph[bias].shape.unwrap_1();
                    let bias_view_shape = Shape::new(vec![Size::ONE, bias_size, Size::ONE, Size::ONE]);

                    let bias_view = graph.view(bias, bias_view_shape);
                    graph.add(conv, bias_view)
                } else {
                    conv
                }
            }
            "Relu" => {
                assert_eq!(1, inputs.len());
                let input = inputs[0];
                graph.relu(input)
            }
            "Clip" => {
                assert_eq!(1, inputs.len());
                let input = inputs[0];
                let min = attrs.take_float("min");
                let max = attrs.take_float("max");

                graph.clamp(input, min, max)
            }
            "Add" => {
                assert_eq!(2, inputs.len());
                let left = inputs[0];
                let right = inputs[1];
                graph.add(left, right)
            }
            "Flatten" => {
                assert_eq!(1, inputs.len());
                let input = inputs[0];

                let rel_axis = attrs.take_int("axis");
                let axis = index_to_abs(rel_axis, graph[input].shape.rank());

                graph.flatten(input, axis)
            }
            "Gemm" => {
                assert_eq!(3, inputs.len());

                let input = inputs[0];
                let weight = inputs[1];
                let bias = inputs.get(2).copied();

                let alpha = attrs.take_float("alpha");
                let beta = attrs.take_float("beta");
                let trans_b = attrs.take_int("transB") != 0;

                assert_eq!(1.0, alpha);
                assert_eq!(1.0, beta);
                assert!(trans_b);

                let [co, ci] = graph[weight].shape.unwrap_fixed().unwrap_2();
                let [n, ci_check] = graph[input].shape.unwrap_2();

                assert_eq!(ci, ci_check.unwrap_fixed(), "Gemm input size and weight mismatch");

                let input_view_shape = Shape::new(vec![n, Size::fixed(ci), Size::ONE, Size::ONE]);
                let input_view = graph.view(input, input_view_shape);

                let filter_view_shape = Shape::fixed(&[co, ci, 1, 1]);
                let filter_view = graph.view(weight, filter_view_shape);

                let conv = graph.conv(input_view, filter_view, 0);

                let output = if let Some(bias) = bias {
                    let co_check = graph[bias].shape.unwrap_1().unwrap_fixed();
                    assert_eq!(co, co_check, "Gemm bias size mismatch");

                    let bias_view_shape = Shape::fixed(&[1, co_check, 1, 1]);
                    let bias_view = graph.view(bias, bias_view_shape);

                    graph.add(conv, bias_view)
                } else {
                    conv
                };

                let output_shape = Shape::new(vec![n, Size::fixed(co)]);
                graph.view(output, output_shape)
            }
            "BatchNormalization" => {
                assert_eq!(5, inputs.len());

                let input = inputs[0];

                // assume everything is constant for now, so we can immediately fuse stuff
                let scale = graph.unwrap_const(inputs[1]);
                let bias = graph.unwrap_const(inputs[2]);
                let mean = graph.unwrap_const(inputs[3]);
                let variance = graph.unwrap_const(inputs[4]);

                let epsilon = attrs.take_float("epsilon");
                let _ = attrs.take_float("momentum");

                // figure out the shapes
                let input_shape = &graph[input].shape;
                assert!(input_shape.rank() >= 2, "BN input must have at least rank 2");
                let const_shape = input_shape.all_ones_except(1);

                let channels = input_shape[1].unwrap_fixed();
                assert!(
                    scale.len() == channels && bias.len() == channels &&
                        mean.len() == channels && variance.len() == channels
                );

                // fuse everything into a single scale and bias
                let total_scale = (0..channels)
                    .map(|i| scale[i] / (variance[i] + epsilon))
                    .collect_vec();
                let total_bias = (0..channels)
                    .map(|i| bias[i] - mean[i] / (variance[i] + epsilon))
                    .collect_vec();

                // put everything into the graph
                let total_scale = graph.constant(const_shape.clone(), total_scale);
                let total_bias = graph.constant(const_shape.clone(), total_bias);

                let scaled = graph.mul(input, total_scale);
                graph.add(scaled, total_bias)
            }
            "Constant" => {
                assert!(inputs.is_empty());

                let tensor = attrs.take_tensor("value");
                let (shape, data) = load_tensor_float_data(tensor);

                println!("Loaded constant with shape {:?} and data {:?}", shape, data);

                graph.constant(shape, data)
            }
            "Cast" => {
                assert_eq!(1, inputs.len());
                let input = inputs[0];

                let to_type = DataType::from_i32(attrs.take_int("to") as i32)
                    .expect("Invalid data type");
                assert_eq!(to_type, DataType::Int64);

                graph.unwrap_const(input);

                // we don't actually cast anything here, and casting is just up to the user
                //  just make sure we're casting a const to int so nothing can go terribly wrong
                input
            }
            "Reshape" => {
                assert_eq!(2, inputs.len());

                let input = inputs[0];
                let new_shape = inputs[1];

                assert_eq!(1, graph[new_shape].shape.rank(), "Reshape shape must have rank 1");
                let new_shape_f = graph.unwrap_const(new_shape);

                let input_shape = &graph[input].shape;
                let output_shape = calculate_reshape_output_shape(input_shape.size(), new_shape_f);

                graph.view(input, output_shape)
            }
            "Gather" => {
                assert_eq!(2, inputs.len());

                let input = inputs[0];
                let indices = inputs[1];

                let axis = attrs.take_int("axis") as usize;

                assert_eq!(graph[indices].shape.rank(), 0, "Only single index gather supported for now");
                let index = graph.unwrap_const(indices)[0] as usize;

                graph.index(input, axis, index)
            }
            "Slice" => {
                assert_eq!(1, inputs.len());
                let input = inputs[0];

                let axes = attrs.take_ints("axes");
                let starts = attrs.take_ints("starts");
                let ends = attrs.take_ints("ends");

                assert!(axes.len() == starts.len() && axes.len() == ends.len(), "Inconsistent axes count");

                (0..axes.len()).fold(input, |curr, i| {
                    graph.slice(curr, axes[i] as usize, starts[i] as usize, ends[i] as usize)
                })
            }
            _ => {
                eprintln!("Already parsed graph:\n{:?}", graph);
                panic!("Unsupported op_type '{}' in node {}", node.op_type, node.name);
            }
        };

        // register operation output as node
        nodes.define(&output_name, value);
        assert!(attrs.is_done(), "Leftover attributes: {:?}", attrs);
    }

    for output in &model_graph.output {
        graph.output(nodes[&output.name]);
    }

    graph
}

fn load_initializers<'a>(graph: &mut Graph, store: &mut Store<'a, Value>, initializers: &'a [TensorProto]) {
    for tensor in initializers {
        let (shape, data) = load_tensor_float_data(tensor);
        let node = graph.constant(shape, data);
        store.define(&tensor.name, node)
    }
}

fn load_tensor_float_data(tensor: &TensorProto) -> (Shape, Vec<f32>) {
    // figure out the dimension
    let dims = tensor.dims.iter().map(|&d| Size::fixed(d as usize)).collect_vec();
    let shape = Shape::new(dims);
    let size = shape.size().unwrap_fixed();

    // load the data
    let data_type = DataType::from_i32(tensor.data_type).expect("Illegal data type");

    let data = match data_type {
        DataType::Float => {
            if !tensor.float_data.is_empty() {
                tensor.float_data.clone()
            } else {
                let mut float_data = vec![0.0; size];
                LittleEndian::read_f32_into(&tensor.raw_data, &mut float_data);
                float_data
            }
        }
        DataType::Double => {
            if !tensor.double_data.is_empty() {
                tensor.double_data.iter().map(|&f| f as f32).collect_vec()
            } else {
                let mut double_data = vec![0.0; size];
                LittleEndian::read_f64_into(&tensor.raw_data, &mut double_data);
                double_data.iter().map(|&f| f as f32).collect_vec()
            }
        }
        _ => panic!("Unexpected data type {:?} {}", data_type, tensor.data_type),
    };

    (shape, data)
}

fn resolve_tensor_shape(ty: &TypeProto) -> Shape {
    let value = ty.value.as_ref().expect("Value doesn't have type set");
    match value {
        ProtoTypeValue::TensorType(tensor) => {
            assert_eq!(tensor.elem_type, DataType::Float as i32, "only floats supported for now");

            let dims = tensor.shape.as_ref()
                .expect("Tensor does not have shape set")
                .dim.iter()
                .map(|d| resolve_tensor_dim(d))
                .collect_vec();
            Shape::new(dims)
        }
        _ => panic!("Unsupported value kind {:?}", value),
    }
}

fn resolve_tensor_dim(dim: &tensor_shape_proto::Dimension) -> Size {
    let value = dim.value.as_ref()
        .expect("Missing value for dimension");

    match value {
        &ProtoDimValue::DimValue(inner) => Size::fixed(inner as usize),
        ProtoDimValue::DimParam(name) => {
            assert_eq!(name, "batch_size");
            Size::BATCH
        }
    }
}

fn index_to_abs(index: i64, size: usize) -> usize {
    if index < 0 {
        size - ((-index) as usize)
    } else {
        index as usize
    }
}

#[track_caller]
fn unwrap_2(slice: &[i64]) -> [usize; 2] {
    assert_eq!(slice.len(), 2, "Expected 2 elements, got {:?}", slice);
    [slice[0] as usize, slice[1] as usize]
}

#[track_caller]
fn unwrap_4(slice: &[i64]) -> [usize; 4] {
    assert_eq!(slice.len(), 4, "Expected 4 elements, got {:?}", slice);
    [slice[0] as usize, slice[1] as usize, slice[2] as usize, slice[3] as usize]
}

fn load_model_proto(path: &Path) -> ModelProto {
    let bytes = std::fs::read(path)
        .unwrap();

    let mut bytes: &[u8] = &bytes;
    let model = ModelProto::decode(&mut bytes)
        .unwrap();

    model
}

fn calculate_reshape_output_shape(old_size: Size, new_shape_f: &[f32]) -> Shape {
    let mut new_shape = vec![];
    let mut leftover_index = None;
    let mut leftover_size = old_size;

    for (i, &size_f) in new_shape_f.iter().enumerate() {
        let size = size_f as i64;
        assert_eq!(size as f32, size_f, "Size must be an integer");

        let size = if size == -1 {
            assert!(leftover_index.is_none(), "Reshape shape can only contain a single -1 value");
            leftover_index = Some(i);
            Size::ZERO
        } else {
            assert!(size >= 0, "Size must be positive or -1");
            let size = Size::fixed(size as usize);
            leftover_size = leftover_size / size;
            size
        };

        new_shape.push(size);
    }

    if let Some(leftover_index) = leftover_index {
        new_shape[leftover_index] = leftover_size;
    }

    let shape = Shape::new(new_shape);
    assert_eq!(old_size, shape.size(), "Output and input sizes differ");

    shape
}
