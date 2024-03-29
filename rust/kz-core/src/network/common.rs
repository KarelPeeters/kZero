use std::borrow::{Borrow, Cow};

use board_game::board::Board;
use board_game::pov::ScalarPov;
use board_game::wdl::WDL;
use internal_iterator::InternalIterator;
use kn_graph::graph::Graph;
use kn_graph::shape;
use kn_graph::shape::{Shape, Size};
use ndarray::{s, ArrayView1, ArrayView2};

use crate::mapping::{BoardMapper, PolicyMapper};
use crate::network::ZeroEvaluation;
use crate::zero::values::ZeroValuesPov;

pub fn decode_output<B: Board, P: PolicyMapper<B>>(
    policy_mapper: P,
    boards: &[impl Borrow<B>],
    outputs: &[&[f32]],
) -> Vec<ZeroEvaluation<'static>> {
    let batch_size = boards.len();
    let policy_len = policy_mapper.policy_len();

    // stack storage for some intermediate values
    let scalars;
    const NAN_SLICE: &[f32] = &[f32::NAN];
    let nan_array = ArrayView1::from(NAN_SLICE);

    // interpret outputs
    let (batch_value_logit, batch_wdl_logit, batch_moves_left, batch_policy_logit) = match outputs.len() {
        2 => {
            scalars = ArrayView2::from_shape((batch_size, 5), outputs[0]).unwrap();
            let policy = ArrayView2::from_shape((batch_size, policy_len), outputs[1]).unwrap();

            (
                scalars.slice(s![.., 0]),
                scalars.slice(s![.., 1..4]),
                scalars.slice(s![.., 4]),
                policy,
            )
        }
        3 => {
            let value = ArrayView1::from_shape(batch_size, outputs[0]).unwrap();
            let wdl = ArrayView2::from_shape((batch_size, 3), outputs[1]).unwrap();
            let moves_left = nan_array.broadcast(batch_size).unwrap();
            let policy = ArrayView2::from_shape((batch_size, policy_len), outputs[2]).unwrap();

            (value, wdl, moves_left, policy)
        }
        _ => unreachable!("Output count should have been checked already"),
    };

    boards
        .iter()
        .enumerate()
        .map(|(bi, board)| {
            let board = board.borrow();

            // simple scalars
            let value = batch_value_logit[bi].tanh();
            let moves_left = batch_moves_left[bi];

            // wdl
            let mut wdl = [
                batch_wdl_logit[(bi, 0)],
                batch_wdl_logit[(bi, 1)],
                batch_wdl_logit[(bi, 2)],
            ];
            softmax_in_place(&mut wdl);
            let wdl = WDL {
                win: wdl[0],
                draw: wdl[1],
                loss: wdl[2],
            };

            // policy
            let policy = board.available_moves().map_or(vec![], |moves| {
                let mut policy: Vec<f32> = moves
                    .map(|mv| {
                        let index = policy_mapper.move_to_index(board, mv);
                        batch_policy_logit[(bi, index)]
                    })
                    .collect();
                softmax_in_place(&mut policy);
                policy
            });

            // combine everything
            let values = ZeroValuesPov {
                value: ScalarPov::new(value),
                wdl,
                moves_left,
            };
            ZeroEvaluation {
                values,
                policy: Cow::Owned(policy),
            }
        })
        .collect()
}

pub fn softmax_in_place(slice: &mut [f32]) {
    let max = slice.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    let mut sum = 0.0;
    for v in slice.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    assert!(sum > 0.0, "Softmax input sum must be strictly positive, was {}", sum);
    for v in slice.iter_mut() {
        *v /= sum;
    }
}

pub fn unsoftmax_in_place(slice: &mut [f32], bias: f32) {
    for v in slice.iter_mut() {
        *v = v.ln() + bias
    }
}

pub fn softmax(slice: &[f32]) -> Vec<f32> {
    let mut copy = slice.to_vec();
    softmax_in_place(&mut copy);
    copy
}

pub fn normalize_in_place(slice: &mut [f32]) {
    let total = slice.iter().sum::<f32>();
    slice.iter_mut().for_each(|f| *f /= total);
}

pub fn policy_softmax_temperature_in_place(slice: &mut [f32], temperature: f32) {
    if temperature == 1.0 {
        return;
    }

    assert!(
        temperature > 0.0 && temperature.is_finite(),
        "Temperature must be finite and positive, got {}",
        temperature
    );

    let mut prev_sum = 0.0;
    let mut sum = 0.0;

    for v in slice.iter_mut() {
        prev_sum += *v;
        *v = v.powf(1.0 / temperature);
        sum += *v;
    }

    assert!(
        0.99 < prev_sum && prev_sum < 1.01,
        "Expected sum 1.0, got {} from {:?}",
        prev_sum,
        slice
    );

    for v in slice {
        *v /= sum;
    }
}

pub fn check_graph_shapes<B: Board, M: BoardMapper<B>>(mapper: M, graph: &Graph) {
    // input
    let inputs = graph.inputs();
    assert_eq!(1, inputs.len(), "Wrong number of inputs");

    let graph_input_shape = &graph[inputs[0]].shape;
    let mapper_input_shape = shape![Size::BATCH].concat(&Shape::fixed(&mapper.input_full_shape()));
    assert_eq!(
        graph_input_shape, &mapper_input_shape,
        "Input shape mismatch between graph and mapper"
    );

    // outputs
    let outputs = graph.outputs();
    let expected_policy_shape = shape![Size::BATCH].concat(&Shape::fixed(mapper.policy_shape()));

    match outputs.len() {
        2 => {
            assert_eq!(&graph[outputs[0]].shape, &shape![Size::BATCH, 5], "Wrong scalars shape");
            assert_eq!(&graph[outputs[1]].shape, &expected_policy_shape, "Wrong policy shape");
        }
        3 => {
            assert_eq!(&graph[outputs[0]].shape, &shape![Size::BATCH], "Wrong value shape");
            assert_eq!(&graph[outputs[1]].shape, &shape![Size::BATCH, 3], "Wrong wdl shape");
            assert_eq!(&graph[outputs[2]].shape, &expected_policy_shape, "Wrong policy shape");
        }
        len => {
            panic!(
                "Wrong number of outputs, expected either (value, wdl, policy) or (scalars, policy), got {}",
                len
            );
        }
    }
}

pub fn zero_values_from_scalars(scalars: &[f32]) -> ZeroValuesPov {
    assert_eq!(scalars.len(), 5, "Expected 5 scalars, got len {}", scalars.len());

    let value = scalars[0].tanh();

    let mut wdl = [scalars[1], scalars[2], scalars[3]];
    softmax_in_place(&mut wdl);

    let moves_left = scalars[4];

    ZeroValuesPov {
        value: ScalarPov::new(value),
        wdl: WDL::new(wdl[0], wdl[1], wdl[2]),
        moves_left,
    }
}
