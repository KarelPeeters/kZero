use itertools::chain;
use ndarray::{Array1, ArrayView1, IxDyn};
use num_traits::real::Real;

use crate::cpu::Tensor;
use crate::graph::{ConvDetails, Graph, Operation, Value};
use crate::optimizer::Optimizer;
use crate::shape::{Shape, Size};

//TODO support any number of trailing dimensions here, specifically for the non-2D case
//TODO or maybe change conv to support any number of trailing dimensions instead
//  so graph doesn't contain a bunch of views?
//TODO support see-though views (when the number of trailing 1 dimensions changes)

//TODO consider fusing the input bias for padding cases as well, by expanding the bias to a full 3D (constant) tensor
//  check which option is the fastest

impl Optimizer<'_> {
    pub fn try_build_affine_group(&self, old_start: Value) -> Option<AffineGroup> {
        let output_shape = &self.old_graph[old_start].shape;

        if let &[batch, after_channels, width, height] = output_shape.dims.as_slice() {
            let after_channels = after_channels.try_unwrap_fixed()?;

            let initial_shape = AffineShape {
                batch,
                before_channels: after_channels,
                after_channels,
                width,
                height,
            };
            let mut builder = AffineGroupBuilder::new(initial_shape);

            let old_input = self.follow_if(old_start, |_, _, operation| {
                self.grow_affine_group(&mut builder, operation)
            });

            if let Some(old_input) = old_input {
                return Some(builder.finish(old_input));
            }
        }

        None
    }

    fn grow_affine_group(&self, builder: &mut AffineGroupBuilder, operation: &Operation) -> Option<Value> {
        match operation {
            &Operation::Conv { input, filter, details: conv_shape } => {
                if let Some(filter) = self.follow_const(filter) {
                    if builder.conv.is_none() && conv_shape.output_size == conv_shape.input_size {
                        builder.set_conv(ConvOperation { details: conv_shape, filter: filter.to_owned() });
                        Some(input)
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            &Operation::Add { left, right, subtract } => {
                if let &[Size::ONE, channels, Size::ONE, Size::ONE] = self.old_graph[right].shape.dims.as_slice() {
                    assert_eq!(channels, Size::fixed(builder.current_channels()));

                    if let Some(data) = self.follow_const(right) {
                        builder.push_affine(AffineOperation::AddChannel { data: data.to_owned(), subtract });
                        Some(left)
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            &Operation::Mul { left, right } => {
                if let &[Size::ONE, channels, Size::ONE, Size::ONE] = self.old_graph[right].shape.dims.as_slice() {
                    assert_eq!(channels, Size::fixed(builder.current_channels()));

                    if let Some(data) = self.follow_const(right) {
                        builder.push_affine(AffineOperation::ScaleChannel { data: data.to_owned() });
                        Some(left)
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

#[derive(Debug)]
struct AffineGroupBuilder {
    shape: AffineShape,

    conv: Option<ConvOperation>,

    before_rev: Vec<AffineOperation>,
    after_rev: Vec<AffineOperation>,
}

impl AffineGroupBuilder {
    fn new(initial_shape: AffineShape) -> Self {
        AffineGroupBuilder {
            shape: initial_shape,
            conv: None,
            before_rev: vec![],
            after_rev: vec![],
        }
    }

    fn current_channels(&self) -> usize {
        // after a conv is set this is overwritten, before that it's initialized to after_channels
        self.shape.before_channels
    }

    fn set_conv(&mut self, conv: ConvOperation) {
        assert!(self.conv.is_none());
        self.shape.before_channels = conv.details.input_channels;
        self.conv = Some(conv);
    }

    fn push_affine(&mut self, operation: AffineOperation) {
        let target = if self.conv.is_some() { &mut self.before_rev } else { &mut self.after_rev };
        target.push(operation);
    }

    fn finish(self, old_input: Value) -> AffineGroup {
        AffineGroup {
            shape: self.shape,
            old_input,
            before: reversed(self.before_rev),
            conv: self.conv,
            after: reversed(self.after_rev),
        }
    }
}

#[derive(Debug)]
pub struct AffineGroup {
    shape: AffineShape,
    old_input: Value,

    before: Vec<AffineOperation>,
    conv: Option<ConvOperation>,
    after: Vec<AffineOperation>,
}

impl AffineGroup {
    pub fn old_input(&self) -> Value {
        self.old_input
    }

    // TODO write unit test for all of these cases
    //  (with some automatic brute forcing of all possibilities)
    pub fn apply_fused(self, graph: &mut Graph, input: Value) -> Value {

        // output:
        // ((bias? conv) | scale)? bias?

        // println!("{:#?}", self);
        // println!("turned into: ");
        // println!("scale: {:?}", before_scale);
        // println!("bias: {:?}", before_bias);
        // println!("conv: {:?}", self.conv);
        // println!("scale: {:?}", after_scale);
        // println!("bias: {:?}", after_bias);

        if let Some(conv) = self.conv {
            let before = fuse_affine_list(self.shape.before_channels, &self.before);
            let after = fuse_affine_list(self.shape.after_channels, &self.after);

            let details = conv.details;

            let mut filter = Tensor::from_shape_vec(
                IxDyn(&details.kernel_shape()),
                conv.filter,
            ).unwrap();

            if details.padding == 0 {
                // we can pull everything, (including the bias) into or through the conv
                todo!()
            }

            match before.try_to_bias_first() {
                Ok(before_flipped) => {
                    // we can fuse the before scale into the filter
                    todo!()
                }
                Err(before) => {
                    // we can't fuse before into the filter
                    todo!()
                }
            }
        }

        // if there is no convolution, everything can be fused into at most a single scale and bias,
        // which is what the fallback will end up doing
        self.apply_fallback(graph, input)
    }

    /// Fallback implementation, should always be correct but maybe not optimal
    fn apply_fallback(self, graph: &mut Graph, input: Value) -> Value {
        let before = fuse_affine_list(self.shape.before_channels, &self.before);
        let after = fuse_affine_list(self.shape.after_channels, &self.after);

        let mut curr = input;

        curr = before.apply(graph,  curr);
        if let Some(conv) = self.conv {
            let filter = graph.constant(Shape::fixed(&conv.details.kernel_shape()), conv.filter);
            curr = graph.conv(curr, filter, conv.details.padding);
        }
        curr = after.apply(graph, curr);

        curr
    }
}

struct ScaleBias {
    scale: Array1<f32>,
    bias: Array1<f32>,
    scale_first: bool,
}

const FLIP_SCALE_TOLERANCE: f32 = 0.001;

impl ScaleBias {
    fn apply(self, graph: &mut Graph, input: Value) -> Value {
        let const_shape = Shape::fixed(&[1, self.scale.len(), 1, 1]);
        let scale = graph.constant(const_shape.clone(), self.scale.to_vec());
        let bias = graph.constant(const_shape.clone(), self.bias.to_vec());

        let mut curr = input;

        if self.scale_first {
            curr = graph.mul(curr, scale);
            curr = graph.add(curr, bias);
        } else {
            curr = graph.add(curr, bias);
            curr = graph.mul(curr, scale);
        }

        curr
    }

    fn try_to_bias_first(self) -> Result<ScaleBias, ScaleBias> {
        assert!(self.scale_first);

        // if the scale is ever close to zero this is undefined or numerically unstable, so don't do the transformation
        if self.scale.iter().any(|&x| x.abs() < FLIP_SCALE_TOLERANCE) {
            return Err(self);
        }

        Ok(ScaleBias {
            bias: self.bias / &self.scale,
            scale: self.scale,
            scale_first: false,
        })
    }
}

fn fuse_affine_list<'a>(channels: usize, operations: impl IntoIterator<Item=&'a AffineOperation>) -> ScaleBias {
    let mut total_scale = Array1::ones(channels);
    let mut total_bias = Array1::zeros(channels);

    for op in operations {
        match op {
            AffineOperation::AddChannel { data, subtract } => {
                let data = ArrayView1::from_shape(channels, data).unwrap();
                if *subtract {
                    total_bias -= &data;
                } else {
                    total_bias += &data;
                }
            }
            AffineOperation::ScaleChannel { data } => {
                let data = ArrayView1::from_shape(channels, data).unwrap();

                total_scale *= &data;
                total_bias *= &data;
            }
        }
    }

    ScaleBias {
        scale: total_scale,
        bias: total_bias,
        scale_first: true,
    }
}

#[derive(Debug)]
struct ConvOperation {
    details: ConvDetails,
    filter: Vec<f32>,
}

#[derive(Debug)]
struct AffineShape {
    batch: Size,
    before_channels: usize,
    after_channels: usize,
    width: Size,
    height: Size,
}

#[derive(Debug)]
enum AffineOperation {
    AddChannel { data: Vec<f32>, subtract: bool },
    ScaleChannel { data: Vec<f32> },
}

fn reversed<T>(mut v: Vec<T>) -> Vec<T> {
    v.reverse();
    v
}