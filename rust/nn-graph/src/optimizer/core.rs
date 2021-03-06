use std::collections::HashMap;

use itertools::Itertools;

use crate::graph::{BinaryOp, Graph, Operation, ReduceOp, UnaryOp, Value};
use crate::optimizer::OptimizerSettings;

#[derive(Debug)]
pub struct Optimizer<'a> {
    settings: OptimizerSettings,

    pub old_graph: &'a Graph,
    pub new_graph: Graph,

    mapping: HashMap<Value, Value>,
}

impl<'a> Optimizer<'a> {
    pub fn new(settings: OptimizerSettings, old_graph: &'a Graph) -> Self {
        Optimizer {
            settings,
            new_graph: Graph::new(),
            old_graph,
            mapping: HashMap::default(),
        }
    }

    pub fn define(&mut self, old: Value, new: Value) {
        let prev = self.mapping.insert(old, new);
        assert!(prev.is_none());
    }

    pub fn map(&mut self, old_value: Value) -> Value {
        if let Some(new_value) = self.mapping.get(&old_value).copied() {
            return new_value;
        }

        let new_value = self.map_new(old_value);
        self.define(old_value, new_value);
        new_value
    }

    fn map_new(&mut self, old_value: Value) -> Value {
        // try fusing the value
        if let Some(fused) = self.try_fuse(old_value) {
            self.new_graph
                .set_debug_id(fused, self.old_graph[old_value].debug_id.clone());
            return fused;
        }

        // fallback, copy the old operation
        let old_info = &self.old_graph[old_value];
        let shape = old_info.shape.clone();

        let old_operation = &old_info.operation;
        let new_operation = old_operation.clone_map_inputs(|old_input| self.map(old_input));

        let new_value = self.new_graph.push(shape, new_operation);
        self.new_graph.set_debug_id(new_value, old_info.debug_id.clone());
        new_value
    }

    fn try_fuse(&mut self, old_start: Value) -> Option<Value> {
        if self.settings.fuse_layernorm {
            if let Some(result) = self.try_fuse_layernorm(old_start) {
                return Some(result);
            }
        }
        if let Some(result) = self.try_fuse_clamp(old_start) {
            return Some(result);
        }
        if let Some(result) = self.try_fuse_conv_affine(old_start) {
            return Some(result);
        }
        if let Some(result) = self.try_convert_div_to_mul(old_start) {
            return Some(result);
        }

        None
    }

    fn try_fuse_layernorm(&mut self, value: Value) -> Option<Value> {
        let mut fused_values = HashMap::<Value, usize>::new();

        let mut op = |v| {
            *fused_values.entry(v).or_insert(0) += 1;
            &self.old_graph[v].operation
        };

        // TODO this is extremely tedious, is there a better way to do general graph matching?
        if let &Operation::Binary {
            left: zeroed0,
            right: std_broadcast,
            op: BinaryOp::Div,
        } = op(value)
        {
            if let &Operation::Binary {
                left: input0,
                right: mean_broadcast,
                op: BinaryOp::Sub,
            } = op(zeroed0)
            {
                if let &Operation::Broadcast { input: mean_view } = op(mean_broadcast) {
                    if let &Operation::View { input: mean } = op(mean_view) {
                        if let &Operation::Reduce {
                            input: input1,
                            axes: ref axes0,
                            op: ReduceOp::Mean,
                        } = op(mean)
                        {
                            if let &Operation::Broadcast { input: std } = op(std_broadcast) {
                                if let &Operation::Unary {
                                    input: stable_var,
                                    op: UnaryOp::Sqrt,
                                } = op(std)
                                {
                                    if let &Operation::Binary {
                                        left: var_view,
                                        right: const_eps,
                                        op: BinaryOp::Add,
                                    } = op(stable_var)
                                    {
                                        if let &Operation::View { input: var } = op(var_view) {
                                            if let &Operation::Reduce {
                                                input: pow,
                                                axes: ref axes1,
                                                op: ReduceOp::Mean,
                                            } = op(var)
                                            {
                                                if let &Operation::Binary {
                                                    left: zeroed1,
                                                    right: const_2,
                                                    op: BinaryOp::Pow,
                                                } = op(pow)
                                                {
                                                    if input0 != input1 || zeroed0 != zeroed1 || axes0 != axes1 {
                                                        return None;
                                                    }

                                                    // check that the intermediate values are not used elsewhere
                                                    if fused_values.iter().any(|(&fused_value, &count)| {
                                                        value != value
                                                            && !self.old_graph.is_hidden_with_users(fused_value, count)
                                                    }) {
                                                        return None;
                                                    }

                                                    return self
                                                        .try_fuse_layernorm_inner(input0, axes0, const_2, const_eps);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        None
    }

    fn try_fuse_layernorm_inner(
        &mut self,
        old_input: Value,
        axes: &[usize],
        old_const_2: Value,
        old_const_eps: Value,
    ) -> Option<Value> {
        // confirm that this is actually a layernorm
        // TODO support multiple layernorm axes?
        if axes.len() != 1 {
            return None;
        }
        let axis = axes[0];

        let eps = self.old_graph.as_single_const(old_const_eps)?;

        if !self.old_graph.is_const_filled_with(old_const_2, 2.0) {
            return None;
        }

        // finally, we can just construct the new layernorm operation
        let new_input = self.map(old_input);
        Some(self.new_graph.layernorm(new_input, axis, eps))
    }

    /// Fuse _multiple_ sequential min and max operations into a single min and max operation.
    fn try_fuse_clamp(&mut self, old_start: Value) -> Option<Value> {
        let mut total_min = f32::NEG_INFINITY;
        let mut total_max = f32::INFINITY;

        let old_input = self.follow_if(old_start, |_, _, operation| {
            if let &Operation::Binary {
                left: old_left,
                right: old_right,
                op: op @ (BinaryOp::Min | BinaryOp::Max),
            } = operation
            {
                // if right is a single constant value we can fuse it
                if let Some(value) = self.old_graph.as_const(old_right) {
                    if value.len() == 1 {
                        let &f = value.iter().next().unwrap();

                        match op {
                            BinaryOp::Min => total_max = f32::min(total_max, f),
                            BinaryOp::Max => total_min = f32::max(total_min, f),
                            _ => unreachable!(),
                        }
                        return Some(old_left);
                    }
                }
            }
            None
        })?;

        let new_input = self.map(old_input);
        let new_output = self.new_graph.clamp(new_input, total_min, total_max);
        Some(new_output)
    }

    // TODO also get this to work for 1D convolutions
    fn try_fuse_conv_affine(&mut self, old_start: Value) -> Option<Value> {
        let group = self.try_build_affine_group(old_start)?;

        let new_input = self.map(group.old_input());
        let new_start = group.apply_fused(self.settings, &mut self.new_graph, new_input);

        Some(new_start)
    }

    fn try_convert_div_to_mul(&mut self, old_start: Value) -> Option<Value> {
        if let &Operation::Binary {
            left,
            right,
            op: BinaryOp::Div,
        } = &self.old_graph[old_start].operation
        {
            if let Some(data) = self.old_graph.as_const(right) {
                let new_data = data.iter().map(|&x| 1.0 / x).collect_vec();
                let new_right = self.new_graph.constant(self.old_graph[right].shape.clone(), new_data);

                let new_left = self.map(left);
                let result = self.new_graph.mul(new_left, new_right);
                Some(result)
            } else {
                None
            }
        } else {
            None
        }
    }

    pub fn follow_if(
        &self,
        start: Value,
        mut next: impl FnMut(&Graph, Value, &Operation) -> Option<Value>,
    ) -> Option<Value> {
        let mut curr = start;

        loop {
            if !self.old_graph.is_hidden_with_users(curr, 1) {
                break;
            }

            if let Some(next) = next(self.old_graph, curr, &self.old_graph[curr].operation) {
                curr = next;
            } else {
                break;
            }
        }

        if curr == start {
            None
        } else {
            Some(curr)
        }
    }
}
