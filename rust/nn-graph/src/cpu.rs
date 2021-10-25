use std::fmt::{Debug, Formatter};
use std::time::Instant;

use bytemuck::cast_slice;
use indexmap::IndexMap;
use itertools::Itertools;
use ndarray::{ArcArray, Array4, ArrayView4, IxDyn, SliceInfo, SliceInfoElem};
use unwrap_match::unwrap_match;

use crate::graph::{BinaryOp, ConstantData, ConvShape, Graph, Operation, Value, ValueInfo};

pub type TensorF = ArcArray<f32, IxDyn>;
pub type TensorI = ArcArray<i32, IxDyn>;

#[derive(Debug, Clone)]
pub enum Tensor {
    F32(TensorF),
    I32(TensorI),
}

pub fn cpu_execute_graph(graph: &Graph, batch_size: usize, inputs: &[&Tensor]) -> ExecutionInfo {
    assert_eq!(graph.inputs().len(), inputs.len(), "Wrong input count");

    let mut map: IndexMap<Value, CalculatedValue> = IndexMap::default();

    for output in graph.values() {
        //TODO handle/check types here
        let ValueInfo { shape, ty: _, operation } = &graph[output];
        let output_shape = shape.eval(batch_size);
        let output_shape_dyn = IxDyn(&output_shape.dims);

        let start_time = Instant::now();

        let result: Tensor = match operation {
            //TODO handle/check types here
            &Operation::Input { ty: _, index } => {
                inputs[index].clone()
            }
            Operation::Constant { data } => {
                Tensor::from_data(output_shape_dyn, data.clone())
            }
            &Operation::View { input } => {
                let input = &map.get(&input).unwrap().tensor;
                match input {
                    Tensor::F32(input) => Tensor::F32(input.reshape(output_shape_dyn)),
                    Tensor::I32(input) => Tensor::I32(input.reshape(output_shape_dyn)),
                }
            }
            &Operation::Slice { input, axis, start, end, } => {
                let input = &map.get(&input).unwrap().tensor;
                let info = slice_info(input.rank(), axis, start, end);

                match input {
                    Tensor::F32(input) => Tensor::F32(input.slice(info).to_shared()),
                    Tensor::I32(input) => Tensor::I32(input.slice(info).to_shared()),
                }
            }
            &Operation::Gather { .. } => todo!(),
            &Operation::Conv { input, filter, conv_shape } => {
                let input = map.get(&input).unwrap().tensor.unwrap_f32()
                    .view().into_dimensionality().unwrap();
                let filter = map.get(&filter).unwrap().tensor.unwrap_f32()
                    .view().into_dimensionality().unwrap();

                let result = convolution(conv_shape, input, filter);
                Tensor::F32(result.into_dyn().into_shared())
            }
            &Operation::Binary { left, right, op } => {
                let left = map.get(&left).unwrap().tensor.unwrap_f32();
                let right = map.get(&right).unwrap().tensor.unwrap_f32();

                let result = match op {
                    BinaryOp::Add => left + right,
                    BinaryOp::Sub => left - right,
                    BinaryOp::Mul => left * right,
                };

                Tensor::F32(result.into_shared())
            }
            &Operation::Clamp { input, min, max } => {
                let input = map.get(&input).unwrap().tensor.unwrap_f32();
                let result = input.map(|&x| x.clamp(min, max));
                Tensor::F32(result.into_shared())
            }
        };

        assert_eq!(&output_shape.dims, result.shape(), "Wrong output shape");

        let end_time = Instant::now();
        let calc = CalculatedValue {
            value: output,
            tensor: result,
            time_spent: (end_time - start_time).as_secs_f32(),
        };
        let prev = map.insert(output, calc);
        assert!(prev.is_none());
    }

    ExecutionInfo {
        batch_size,
        values: map,
        outputs: graph.outputs().to_owned(),
    }
}

fn convolution(shape: ConvShape, input: ArrayView4<f32>, filter: ArrayView4<f32>) -> Array4<f32> {
    let kernel_offset = (shape.kernel_size / 2) as isize;
    let input_range = 0..shape.input_size as isize;

    let output_shape = (input.dim().0, shape.output_channels, shape.output_size, shape.output_size);
    Array4::from_shape_fn(output_shape, |(n, co, ox, oy)| {
        let mut result: f32 = 0.0;

        for ci in 0..shape.input_channels {
            for kx in 0..shape.kernel_size as isize {
                for ky in 0..shape.kernel_size as isize {
                    let ix = ox as isize + kx - kernel_offset;
                    let iy = oy as isize + ky - kernel_offset;

                    if input_range.contains(&ix) && input_range.contains(&iy) {
                        let a = input[(n, ci, ix as usize, iy as usize)];
                        let f = filter[(co, ci, kx as usize, ky as usize)];

                        result += a * f
                    }
                }
            }
        }

        result
    })
}

pub fn slice_info(rank: usize, axis: usize, start: usize, end: usize) -> SliceInfo<Vec<SliceInfoElem>, IxDyn, IxDyn> {
    let vec = (0..rank)
        .map(|r| {
            if r == axis {
                // grab the relevant range
                SliceInfoElem::Slice { start: start as isize, end: Some(end as isize), step: 1 }
            } else {
                // grab everything
                SliceInfoElem::Slice { start: 0, end: None, step: 1 }
            }
        })
        .collect_vec();

    // safety: we pass an owned Vec, whose .as_ref will always return the same reference
    unsafe { SliceInfo::new(vec).unwrap() }
}

#[derive(Debug)]
pub struct ExecutionInfo {
    pub batch_size: usize,
    pub values: IndexMap<Value, CalculatedValue>,
    pub outputs: Vec<Value>,
}

pub struct CalculatedValue {
    pub value: Value,
    pub tensor: Tensor,
    pub time_spent: f32,
}

impl ExecutionInfo {
    pub fn output_tensors(self) -> Vec<Tensor> {
        self.outputs.iter()
            .map(|v| {
                // convert to standard layout so users get easily get &[f32] slices
                let output = &self.values.get(v).unwrap().tensor;
                match output {
                    Tensor::F32(output) => Tensor::F32(output.as_standard_layout().to_shared()),
                    Tensor::I32(output) => Tensor::I32(output.as_standard_layout().to_shared()),
                }
            })
            .collect_vec()
    }
}

impl Debug for CalculatedValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CalculatedTensor")
            .field("value", &self.value)
            .field("shape", &self.tensor.shape())
            .field("time_spent", &self.time_spent)
            .finish()
    }
}

impl Tensor {
    pub fn from_data(shape: IxDyn, data: ConstantData) -> Self {
        match data {
            ConstantData::F32(vec) =>
                Tensor::F32(ArcArray::from_shape_vec(shape, vec).unwrap()),
            ConstantData::I32(vec) =>
                Tensor::I32(ArcArray::from_shape_vec(shape, vec).unwrap()),
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            Tensor::F32(inner) => inner.shape(),
            Tensor::I32(inner) => inner.shape(),
        }
    }

    pub fn rank(&self) -> usize {
        self.shape().len()
    }

    pub fn unwrap_f32(&self) -> &ArcArray<f32, IxDyn> {
        unwrap_match!(self, Tensor::F32(tensor) => tensor)
    }

    pub fn to_f32(&self) -> ArcArray<f32, IxDyn> {
        match self {
            Tensor::F32(tensor) => tensor.to_shared(),
            Tensor::I32(tensor) => tensor.mapv(|x| x as f32).to_shared(),
        }
    }

    pub fn as_byte_slice(&self) -> Option<&[u8]> {
        match self {
            Tensor::F32(tensor) => tensor.as_slice().map(cast_slice),
            Tensor::I32(tensor) => tensor.as_slice().map(cast_slice),
        }
    }
}
