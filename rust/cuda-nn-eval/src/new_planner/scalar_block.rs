use std::ops::ControlFlow;

use rand::{thread_rng, Rng};

use decorum::Total;
use nn_graph::graph::{BinaryOp, UnaryOp};

use crate::new_planner::core::Operand;

#[derive(Debug)]
pub struct ScalarBlock<B> {
    check: u64,
    expressions: Vec<Expression<B>>,
    outputs: Vec<(B, ScalarValue)>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Expression<B> {
    Input(B),
    Constant(Total<f32>),
    Unary(UnaryOp, ScalarValue),
    Binary(BinaryOp, ScalarValue, ScalarValue),
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct ScalarValue {
    index: usize,
    check: u64,
}

impl<B: Eq> ScalarBlock<B> {
    pub fn new() -> Self {
        Self {
            check: thread_rng().gen(),
            expressions: vec![],
            outputs: vec![],
        }
    }

    pub fn for_each_operand<R>(&self, mut f: impl FnMut(Operand<&B>) -> ControlFlow<R>) -> ControlFlow<R> {
        for expr in &self.expressions {
            if let Expression::Input(buffer) = expr {
                f(Operand::read(buffer))?;
            }
        }

        for (output, _) in &self.outputs {
            f(Operand::write(output))?;
        }

        ControlFlow::Continue(())
    }

    pub fn check_contains(&self, value: ScalarValue) {
        assert!(
            self.check == value.check && value.index < self.expressions.len(),
            "Value {:?} does not belong to this ScalarBlock",
            value
        );
    }

    pub fn push(&mut self, expr: Expression<B>) -> ScalarValue {
        // TODO faster (hashmap) deduplication
        let index = if let Some(index) = self.expressions.iter().position(|other| other == &expr) {
            index
        } else {
            // check valid operands
            match expr {
                Expression::Input(ref _buffer) => {}
                Expression::Constant(_value) => {}
                Expression::Unary(_op, input) => {
                    self.check_contains(input);
                }
                Expression::Binary(_op, left, right) => {
                    self.check_contains(left);
                    self.check_contains(right);
                }
            }

            // push to expression list
            let index = self.expressions.len();
            self.expressions.push(expr);
            index
        };

        ScalarValue {
            index,
            check: self.check,
        }
    }

    pub fn input(&mut self, buffer: B) -> ScalarValue {
        self.push(Expression::Input(buffer))
    }

    pub fn output(&mut self, buffer: B, value: ScalarValue) {
        self.outputs.push((buffer, value));
    }

    pub fn copy(input: B, output: B) -> Self {
        let mut block = ScalarBlock::new();
        let input_value = block.input(input);
        block.output(output, input_value);
        block
    }

    pub fn unary(op: UnaryOp, input: B, output: B) -> Self {
        let mut block = ScalarBlock::new();
        let input_value = block.input(input);
        let output_value = block.push(Expression::Unary(op, input_value));
        block.output(output, output_value);
        block
    }

    pub fn binary(op: BinaryOp, left: B, right: B, output: B) -> Self {
        let mut block = ScalarBlock::new();
        let left_value = block.input(left);
        let right_value = block.input(right);
        let output_value = block.push(Expression::Binary(op, left_value, right_value));
        block.output(output, output_value);
        block
    }
}
