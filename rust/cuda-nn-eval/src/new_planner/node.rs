use std::ops::ControlFlow;

use crate::new_planner::core::Operand;
use nn_graph::graph::{ConvDetails, ReduceOp};

use crate::new_planner::scalar_block::ScalarBlock;

#[derive(Debug)]
pub enum Node<B> {
    Scalar(ScalarBlock<B>),
    MatMul {
        alpha: f32,
        left: B,
        right: B,
        beta: f32,
        output: B,
    },
    Conv {
        details: ConvDetails,
        input: B,
        filter: B,
        output: B,
        bias: Option<B>,
        res: Option<B>,
    },
    Gather {
        input: B,
        axis: usize,
        indices: B,
        output: B,
    },
    Softmax {
        input: B,
        axis: usize,
        output: B,
    },
    Layernorm {
        input: B,
        axis: usize,
        eps: f32,
        output: B,
    },
    Reduce {
        input: B,
        axes: Vec<usize>,
        op: ReduceOp,
        output: B,
    },
}

impl<B> Node<B> {
    fn for_each_operand<R>(&self, mut f: impl FnMut(Operand<&B>) -> ControlFlow<R>) -> ControlFlow<R> {
        match self {
            Node::Scalar(block) => block.for_each_operand(f)?,
            Node::MatMul {
                alpha: _,
                left,
                right,
                beta: _,
                output,
            } => {
                f(Operand::read(left))?;
                f(Operand::read(right))?;
                f(Operand::write(output))?;
            }
            Node::Conv {
                details: _,
                input,
                filter,
                output,
                bias,
                res,
            } => {
                f(Operand::read(input))?;
                f(Operand::read(filter))?;
                if let Some(bias) = bias {
                    f(Operand::read(bias))?;
                }
                if let Some(res) = res {
                    f(Operand::read(res))?;
                }
                f(Operand::write(output))?;
            }
            Node::Gather {
                input,
                axis: _,
                indices,
                output,
            } => {
                f(Operand::read(input))?;
                f(Operand::read(indices))?;
                f(Operand::write(output))?;
            }
            Node::Softmax { input, axis: _, output } => {
                f(Operand::read(input))?;
                f(Operand::write(output))?;
            }
            Node::Layernorm {
                input,
                axis: _,
                eps: _,
                output,
            } => {
                f(Operand::read(input))?;
                f(Operand::write(output))?;
            }
            Node::Reduce {
                input,
                axes: _,
                op: _,
                output,
            } => {
                f(Operand::read(input))?;
                f(Operand::write(output))?;
            }
        }

        ControlFlow::Continue(())
    }
}
