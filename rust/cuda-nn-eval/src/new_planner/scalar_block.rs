use std::fmt::Write;
use std::fmt::{Debug, Formatter};
use std::ops::ControlFlow;

use decorum::Total;
use rand::{thread_rng, Rng};

use nn_graph::graph::{BinaryOp, UnaryOp};

use crate::autokernel::common::DisplayCFloat;
use crate::new_planner::core::Operand;
use crate::planner::{binary_op_str, unary_op_str};

pub struct ScalarBlock<B> {
    check: u64,
    operands: Vec<ScalarOperand<B>>,
    expressions: Vec<Expression>,
}

#[derive(Debug)]
struct ScalarOperand<B> {
    buffer: B,
    is_input: bool,
    output_value: Option<ScalarValue>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Expression {
    Input(usize),
    // TODO start emitting constant expressions
    Constant(Total<f32>),
    Unary(UnaryOp, ScalarValue),
    Binary(BinaryOp, ScalarValue, ScalarValue),
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub struct ScalarValue {
    index: usize,
    check: u64,
}

impl Debug for ScalarValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "ScalarValue({})", self.index)
    }
}

impl<B> ScalarBlock<B> {
    pub fn new() -> Self {
        Self {
            check: thread_rng().gen(),
            operands: vec![],
            expressions: vec![],
        }
    }

    pub fn for_each_operand<R>(&self, mut f: impl FnMut(Operand<&B>) -> ControlFlow<R>) -> ControlFlow<R> {
        for operand in &self.operands {
            if operand.is_input {
                f(Operand::read(&operand.buffer))?;
            }
            if operand.output_value.is_some() {
                f(Operand::write(&operand.buffer))?;
            }
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

    pub fn push(&mut self, expr: Expression) -> ScalarValue {
        // TODO faster (hashmap) deduplication
        let index = if let Some(index) = self.expressions.iter().position(|other| other == &expr) {
            index
        } else {
            // check valid operands
            match expr {
                Expression::Input(index) => {
                    assert!(index < self.operands.len());
                    self.operands[index].is_input = true;
                }
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

    pub fn to_operation(&self) -> String {
        let mut result = String::new();
        let f = &mut result;

        for (y_expr, &expr) in self.expressions.iter().enumerate() {
            let op_str = match expr {
                Expression::Input(xi) => {
                    format!("*x{xi}")
                }
                Expression::Constant(f) => {
                    let f = *f.as_ref();
                    format!("{}", DisplayCFloat(f))
                }
                Expression::Unary(op, input) => {
                    let y_input = input.index;
                    unary_op_str(op, &format!("y{}", y_input))
                }
                Expression::Binary(op, left, right) => {
                    let y_left = left.index;
                    let y_right = right.index;
                    binary_op_str(op, &format!("y{}", y_left), &format!("y{}", y_right))
                }
            };

            writeln!(f, "y{y_expr} = {op_str};").unwrap();
        }

        for (x_op, operand) in self.operands.iter().enumerate() {
            if let Some(output_value) = operand.output_value {
                let y_output = output_value.index;
                writeln!(f, "*x{x_op} = y{y_output};").unwrap();
            }
        }

        result
    }
}

impl<B: Eq> ScalarBlock<B> {
    fn push_operand(&mut self, buffer: B) -> usize {
        if let Some(index) = self.operands.iter().position(|operand| &operand.buffer == &buffer) {
            index
        } else {
            let operand = ScalarOperand {
                buffer,
                is_input: false,
                output_value: None,
            };
            let index = self.operands.len();
            self.operands.push(operand);
            index
        }
    }

    pub fn input(&mut self, buffer: B) -> ScalarValue {
        let index = self.push_operand(buffer);
        self.push(Expression::Input(index))
    }

    pub fn output(&mut self, buffer: B, value: ScalarValue) {
        let index = self.push_operand(buffer);
        let output_value = &mut self.operands[index].output_value;
        assert!(
            output_value.is_none(),
            "Trying to write output {:?} to buffer {} which already has output {:?}",
            value,
            index,
            output_value
        );

        *output_value = Some(value);
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

impl<B: Debug> Debug for ScalarBlock<B> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScalarBlock")
            .field("operation", &self.to_operation())
            .field("operands", &self.operands)
            .finish()
    }
}
