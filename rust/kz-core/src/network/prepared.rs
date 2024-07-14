use std::borrow::Borrow;
use std::marker::PhantomData;

use board_game::board::Board;
use itertools::Itertools;
use kn_graph::dtype::{DTensor, Tensor};
use kn_graph::graph::Graph;
use kn_runtime::{Device, PreparedGraph};
use ndarray::IxDyn;

use crate::mapping::BoardMapper;
use crate::network::common::decode_output;
use crate::network::{Network, ZeroEvaluation};

#[derive(Debug)]
pub struct PreparedNetwork<B: Board, M: BoardMapper<B>> {
    mapper: M,
    graph: PreparedGraph,
    batch_size: usize,
    ph: PhantomData<B>,
}

impl<B: Board, M: BoardMapper<B>> PreparedNetwork<B, M> {
    pub fn new(mapper: M, device: Device, graph: Graph, batch_size: usize) -> Self {
        Self {
            mapper,
            graph: device.prepare(graph, batch_size),
            batch_size,
            ph: PhantomData,
        }
    }
}

impl<B: Board, M: BoardMapper<B>> Network<B> for PreparedNetwork<B, M> {
    fn max_batch_size(&self) -> usize {
        self.batch_size
    }

    fn evaluate_batch(&mut self, boards: &[impl Borrow<B>]) -> Vec<ZeroEvaluation<'static>> {
        assert!(boards.len() <= self.batch_size);

        // encore the input
        let mut input = vec![];
        for board in boards {
            self.mapper.encode_input_full(&mut input, board.borrow())
        }

        // pad with dummy input
        // TODO skip this if the underlying executor supports this (once Kyanite does)
        for _ in boards.len()..self.batch_size {
            for _ in 0..self.mapper.input_full_len() {
                input.push(0.0);
            }
        }

        // mess with shapes and dtypes
        let mut input_shape = vec![self.batch_size];
        input_shape.extend_from_slice(&self.mapper.input_full_shape());
        let input = Tensor::from_shape_vec(IxDyn(&input_shape), input)
            .unwrap_or_else(|_| panic!("Incompatible shapes: ({}) -> {:?}", self.batch_size, input_shape));

        // evaluate the graph
        let outputs = self.graph.eval(&[DTensor::F32(input)]);

        // unwrap the output types
        let batch_outputs = outputs
            .iter()
            .map(|t| t.unwrap_f32().unwrap().as_slice().unwrap())
            .collect_vec();

        let mut result = decode_output(self.mapper, boards, &batch_outputs);

        // remove padding again
        result.truncate(boards.len());
        result
    }
}
