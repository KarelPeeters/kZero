use std::collections::{HashMap, HashSet, VecDeque};

use itertools::Itertools;

use nn_graph::graph::{ConstantData, Graph, Operation, Value};
use nn_graph::shape::ConcreteShape;

use crate::autokernel::scalar::ScalarKernel;
use crate::executor::Handles;
use crate::new_planner::node::Node;
use crate::new_planner::scalar_block::ScalarBlock;
use crate::offset_tensor::{OffsetPtr, PtrTensor};
use crate::shape::StridedShape;
use crate::step::{ScalarOpArgs, Step};

type NewTensor = PtrTensor<NewPtr>;

pub struct NewPlanner<'a> {
    graph: &'a Graph,
    batch_size: usize,

    nodes: Vec<Node<NewTensor>>,
    inputs: Vec<NewTensor>,
    constants: Vec<(NewTensor, &'a ConstantData)>,

    value_to_tensor: HashMap<Value, NewTensor>,
    buffer_sizes: Vec<usize>,
}

impl<'a> NewPlanner<'a> {
    pub fn plan(handles: &'a Handles, graph: &'a Graph, batch_size: usize) {
        let mut planner = NewPlanner::new(graph, batch_size);

        planner.add_initial_nodes();

        println!("New planner steps:");
        for node in &planner.nodes {
            println!("  {:?}", node);
        }

        println!();
        println!();

        // TODO fuse nodes

        // TODO fuse scalar operations

        // TODO somehow alloc scratch buffers at some point?

        // TODO allocate buffers (optimally)

        // TODO decide on real strides?
        //   where can we do the "transpose(matmul)" optimization?

        // TODO convert everything to a proper plan

        let device = handles.device();

        // TODO is all of this actually the right level of abstraction?
        //   it feels like it's going to be annoying to respect requirements and decide strides after we've split
        //   things into PtrTensors

        let steps = planner
            .nodes
            .iter()
            .map(|node| match node {
                Node::Scalar(block) => {
                    let operands = block
                        .operands()
                        .iter()
                        .map(|operand| operand.buffer.clone())
                        .collect_vec();
                    let shapes = operands
                        .iter()
                        .map(|operand| operand.strided_shape().clone())
                        .collect_vec();

                    let kernel = ScalarKernel::new_for_shapes(device, &block.to_operation(), &shapes);
                    Step::ScalarOp(ScalarOpArgs { kernel, operands })
                }
                &Node::MatMul { .. } => todo!(),
                Node::Conv { .. } => todo!(),
                Node::Gather { .. } => todo!(),
                Node::Softmax { .. } => todo!(),
                Node::Layernorm { .. } => todo!(),
                Node::Reduce { .. } => todo!(),
            })
            .collect_vec();

        todo!("actually plan things");
    }

    fn new(graph: &'a Graph, batch_size: usize) -> Self {
        Self {
            graph,
            batch_size,
            nodes: vec![],
            value_to_tensor: HashMap::new(),
            buffer_sizes: vec![],
            inputs: vec![],
            constants: vec![],
        }
    }

    fn add_initial_nodes(&mut self) {
        let values = collect_used_values(self.graph);

        self.inputs = self.graph.inputs().iter().map(|&v| self.as_tensor(v)).collect_vec();

        for output in values {
            let output_info = &self.graph[output];
            let output_shape = output_info.shape.eval(self.batch_size);
            let output = self.as_tensor(output);

            match output_info.operation {
                // inputs
                Operation::Input { .. } => {
                    // already handled earlier
                }
                Operation::Constant { ref data } => {
                    self.constants.push((output, data));
                }

                // shape/stride trickery
                Operation::View { input } => {
                    let input = self.as_tensor(input);

                    // convert to dense strides with an additional copy
                    let stage = self.alloc_tensor(output_shape.clone());
                    self.push_copy(input, stage.clone());

                    // view operation cannot fail on dense input
                    let stage_viewed = stage.view(output_shape.dims).unwrap();
                    self.push_copy(stage_viewed, output);
                }
                Operation::Broadcast { input } => {
                    let input = self.as_tensor(input);
                    self.push_copy(input.broadcast(output_shape.dims), output)
                }
                Operation::Permute { input, ref permutation } => {
                    let input = self.as_tensor(input);
                    self.push_copy(input.permute(permutation), output);
                }
                Operation::Slice { input, axis, range } => {
                    let input = self.as_tensor(input);
                    self.push_copy(input.slice(axis, range), output);
                }
                Operation::Flip { input, axis } => {
                    let input = self.as_tensor(input);
                    self.push_copy(input.flip(axis), output);
                }
                Operation::Concat { ref inputs, axis } => {
                    let mut curr_start = 0;

                    // copy each input into the corresponding slice of the output
                    for &input in inputs {
                        let input = self.as_tensor(input);

                        let curr_len = input.strided_shape().shape()[axis];
                        let curr_range = curr_start..curr_start + curr_len;

                        let output_slice = output.slice(axis, curr_range);
                        self.push_copy(input, output_slice);

                        curr_start += curr_len;
                    }
                }

                // scalar
                Operation::Unary { input, op } => {
                    let input = self.as_tensor(input);
                    let block = ScalarBlock::unary(op, input, output);
                    self.push_node(Node::Scalar(block));
                }
                Operation::Binary { left, right, op } => {
                    let left = self.as_tensor(left);
                    let right = self.as_tensor(right);
                    let block = ScalarBlock::binary(op, left, right, output);
                    self.push_node(Node::Scalar(block));
                }

                // custom
                Operation::Gather { input, axis, indices } => {
                    let input = self.as_tensor(input);
                    let indices = self.as_tensor(indices);
                    self.push_node(Node::Gather {
                        input,
                        axis,
                        indices,
                        output,
                    });
                }
                Operation::Softmax { input, axis } => {
                    let input = self.as_tensor(input);
                    self.push_node(Node::Softmax { input, axis, output });
                }
                Operation::Layernorm { input, axis, eps } => {
                    let input = self.as_tensor(input);
                    let eps = *eps.as_ref();
                    self.push_node(Node::Layernorm {
                        input,
                        axis,
                        eps,
                        output,
                    });
                }
                Operation::Reduce { input, ref axes, op } => {
                    let input = self.as_tensor(input);
                    let axes = axes.to_vec();
                    self.push_node(Node::Reduce {
                        input,
                        axes,
                        op,
                        output,
                    });
                }

                // cudnn/cublas
                Operation::Conv { details, input, filter } => {
                    let input = self.as_tensor(input);
                    let filter = self.as_tensor(filter);

                    self.push_node(Node::Conv {
                        details,
                        input,
                        filter,
                        bias: None,
                        res: None,
                        output,
                    })
                }
                Operation::MatMul { left, right } => {
                    let left = self.as_tensor(left);
                    let right = self.as_tensor(right);

                    self.push_node(Node::MatMul {
                        alpha: 1.0,
                        left,
                        right,
                        beta: 1.0,
                        output,
                    })
                }
            }
        }
    }

    fn as_tensor(&mut self, value: Value) -> NewTensor {
        if let Some(tensor) = self.value_to_tensor.get(&value) {
            return tensor.clone();
        }

        let shape = self.graph[value].shape.eval(self.batch_size);
        let tensor = self.alloc_tensor(shape);
        self.value_to_tensor.insert(value, tensor.clone());

        tensor
    }

    fn alloc_tensor(&mut self, shape: ConcreteShape) -> NewTensor {
        // TODO update this factor when we get typed tensors
        let buffer = self.alloc_buffer(shape.size() * 4);
        NewTensor::from_parts(buffer, StridedShape::new_simple(shape.dims))
    }

    fn alloc_buffer(&mut self, size_in_bytes: usize) -> NewPtr {
        let index = self.buffer_sizes.len();
        self.buffer_sizes.push(size_in_bytes);
        NewPtr::new(index, 0)
    }

    fn push_node(&mut self, node: Node<NewTensor>) {
        self.nodes.push(node);
    }

    fn push_copy(&mut self, from: NewTensor, to: NewTensor) {
        let block = ScalarBlock::copy(from, to);
        self.push_node(Node::Scalar(block));
    }
}

fn collect_used_values(graph: &Graph) -> HashSet<Value> {
    let mut seen = HashSet::new();
    let mut todo = VecDeque::new();

    todo.extend(graph.inputs().iter().copied());
    todo.extend(graph.outputs().iter().copied());

    while let Some(curr) = todo.pop_front() {
        if seen.insert(curr) {
            todo.extend(graph[curr].operation.inputs().into_iter());
        }
    }

    seen
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct NewPtr {
    index: usize,
    // TODO we need an offset per axis if we want to still allow permuting afterwards
    //   is that enough? -> not really, what about view operations that combine different axes together?
    offset: isize,
}

impl NewPtr {
    fn new(index: usize, offset: isize) -> Self {
        NewPtr { index, offset }
    }
}

impl OffsetPtr for NewPtr {
    fn offset_bytes(self, offset: isize) -> Self {
        NewPtr::new(self.index, self.offset + offset)
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Operand<B> {
    inner: B,
    kind: OperandKind,
}

#[derive(Debug, Copy, Clone)]
pub enum OperandKind {
    Read,
    Write,
}

impl<B> Operand<B> {
    pub fn new(inner: B, kind: OperandKind) -> Self {
        Self { inner, kind }
    }

    pub fn read(inner: B) -> Self {
        Self::new(inner, OperandKind::Read)
    }

    pub fn write(inner: B) -> Self {
        Self::new(inner, OperandKind::Write)
    }

    pub fn kind(&self) -> OperandKind {
        self.kind
    }

    pub fn as_ref(&self) -> Operand<&B> {
        Operand::new(&self.inner, self.kind)
    }
}
