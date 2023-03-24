use crate::cpu::Tensor;
use crate::graph::{BinaryOp, Graph, ReduceOp, UnaryOp, Value};
use crate::shape;
use crate::shape::Size;
use byteorder::{LittleEndian, ReadBytesExt};
use ndarray::IxDyn;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::num::{ParseFloatError, ParseIntError};
use std::path::Path;
use std::str::FromStr;

type Result<T> = std::result::Result<T, Error>;

// TODO remove print statements

pub fn load_katago_network_path(path: impl AsRef<Path>, board_size: usize) -> Result<Graph> {
    let input = BufReader::new(File::open(path)?);
    Loader::load(input, board_size)
}

pub fn load_katago_network<R: Read>(input: BufReader<R>, board_size: usize) -> Result<Graph> {
    Loader::load(input, board_size)
}

#[derive(Debug)]
pub enum Error {
    Eof,
    IO(std::io::Error),

    ParseInt(String, ParseIntError),
    ParseFloat(String, ParseFloatError),
    ParseBool(String),

    InvalidBinPrefix(String),
    UnsupportedBlockKind(String),
    UnsupportedActivationKind(String),
}

struct Loader<R: Read> {
    input: BufReader<R>,
    graph: Graph,
    version: i32,
    board_size: usize,
}

#[derive(Debug)]
enum PoolKind {
    KataGPool,
    KataValueHeadGPool,
}

impl<R: Read> Loader<R> {
    fn load(input: BufReader<R>, board_size: usize) -> Result<Graph> {
        let mut loader = Loader {
            input,
            graph: Graph::new(),
            board_size,
            version: -1,
        };
        loader.load_impl()?;
        Ok(loader.graph)
    }

    fn load_impl(&mut self) -> Result<()> {
        let name = self.read_str()?;
        let version: i32 = self.read_int()?;
        self.version = version;

        println!("Parsing '{}' as network with version {}", name, self.version);

        let channels_in: usize = self.read_int()?;
        let channels_in_global: usize = self.read_int()?;

        let input_spatial = self
            .graph
            .input(shape![Size::BATCH, channels_in, self.board_size, self.board_size]);
        let input_global = self.graph.input(shape![Size::BATCH, channels_in_global]);

        let trunk = self.load_trunk(input_spatial, input_global)?;

        let policy = self.load_policy_head(trunk)?;
        let (value_wdl, value_misc, value_own) = self.load_value_head(trunk)?;

        self.graph.output_all(&[policy, value_wdl, value_misc, value_own]);
        Ok(())
    }

    fn load_trunk(&mut self, input_spatial: Value, input_global: Value) -> Result<Value> {
        let name = self.read_str()?;
        println!("Parsing '{}' as trunk", name);

        let num_blocks: usize = self.read_int()?;
        let _trunk_num_channels: usize = self.read_int()?;
        let _mid_num_channels: usize = self.read_int()?;
        let _regular_num_channels: usize = self.read_int()?;
        let _dilated_num_channels: usize = self.read_int()?;
        let _gpool_num_channels: usize = self.read_int()?;

        let initial_conv = self.load_conv_layer(input_spatial)?;
        let initial_global = self.load_matmul_layer(input_global)?;

        let shape_expanded = self.graph[initial_global].shape.clone().concat(&shape![1, 1]);
        let initial_global_expanded = self.graph.view(initial_global, shape_expanded);
        let start = self.graph.add(initial_conv, initial_global_expanded);

        let result_raw = self.load_residual_stack(start, num_blocks)?;

        let result_norm = self.load_bn_layer(result_raw)?;
        let result = self.load_activation(result_norm)?;

        self.graph.set_debug_id(result, name);
        Ok(result)
    }

    fn load_policy_head(&mut self, trunk: Value) -> Result<Value> {
        let name = self.read_str()?;
        println!("Parsing '{}' as policy head", name);

        let p1_conv = self.load_conv_layer(trunk)?;
        let g1_conv = self.load_conv_layer(trunk)?;

        let g1_norm = self.load_bn_layer(g1_conv)?;
        let g1_act = self.load_activation(g1_norm)?;

        let g1_pooled = self.kata_gpool(g1_act, PoolKind::KataGPool);

        let (squeeze_shape, _) = self.graph[g1_pooled].shape.clone().split(2);

        let g1_squeeze = self.graph.view(g1_pooled, squeeze_shape);

        let gpool_to_bias_mul = self.load_matmul_layer(g1_squeeze)?;
        let unsqueeze_shape = self.graph[gpool_to_bias_mul].shape.clone().concat(&shape![1, 1]);
        let bias = self.graph.view(gpool_to_bias_mul, unsqueeze_shape);

        let out_p = self.graph.add(p1_conv, bias);

        let p1_norm = self.load_bn_layer(out_p)?;
        let p1_act = self.load_activation(p1_norm)?;

        let p2_conv = self.load_conv_layer(p1_act)?;
        let gpool_to_pass_mul = self.load_matmul_layer(g1_squeeze)?;

        let out_policy = self.graph.flatten(p2_conv, 2);
        let pass_unsqueeze = self.graph[gpool_to_pass_mul].shape.clone().concat(&shape![1]);
        let out_pass = self.graph.view(gpool_to_pass_mul, pass_unsqueeze);

        let result = self.graph.concat(vec![out_policy, out_pass], 2, None);

        self.graph.set_debug_id(result, name);
        Ok(result)
    }

    fn load_value_head(&mut self, trunk: Value) -> Result<(Value, Value, Value)> {
        let name = self.read_str()?;
        println!("Parsing '{}' as value head", name);

        let v1_conv = self.load_conv_layer(trunk)?;
        let v1_bn = self.load_bn_layer(v1_conv)?;
        let v1_act = self.load_activation(v1_bn)?;

        let out_pooled = self.kata_gpool(v1_act, PoolKind::KataValueHeadGPool);

        let v2_mul = self.load_matmul_layer(out_pooled)?;
        let v2_bias = self.load_mat_bias_layer(v2_mul)?;
        let v2_act = self.load_activation(v2_bias)?;

        // output shapes:
        println!("v1_act: {:?}", self.graph[v1_act].shape);
        println!("out_pooled: {:?}", self.graph[out_pooled].shape);
        println!("v2_act: {:?}", self.graph[v2_act].shape);

        let v3_mul = self.load_matmul_layer(v2_act)?;
        let v3_bias = self.load_mat_bias_layer(v3_mul)?;

        let sv3_mul = self.load_matmul_layer(v2_act)?;
        let sv3_bias = self.load_mat_bias_layer(sv3_mul)?;

        let ownership_conv = self.load_conv_layer(v1_act)?;

        // value(wdl), misc(score, score_stddev, lead, vtime, short0, short1), ownership

        // print shapes:
        println!("wdl: {:?}", self.graph[v3_bias].shape);
        println!("misc: {:?}", self.graph[sv3_bias].shape);
        println!("ownership: {:?}", self.graph[ownership_conv].shape);

        Ok((v3_bias, sv3_bias, ownership_conv))
    }

    fn load_residual_stack(&mut self, input: Value, num_blocks: usize) -> Result<Value> {
        let mut curr = input;

        for i in 0..num_blocks {
            let kind = self.read_str()?;
            println!("Parsing block {i}/{num_blocks}, kind: {kind}");

            match &*kind {
                "ordinary_block" => curr = self.load_ordinary_block(curr)?,
                "gpool_block" => curr = self.load_gpool_block(curr)?,
                _ => return Err(Error::UnsupportedBlockKind(kind)),
            }
        }

        Ok(curr)
    }

    fn load_ordinary_block(&mut self, input: Value) -> Result<Value> {
        let name = self.read_str()?;
        println!("Parsing '{}' as ordinary_block", name);

        let curr = input;

        let curr = self.load_bn_layer(curr)?;
        let curr = self.load_activation(curr)?;
        let curr = self.load_conv_layer(curr)?;

        let curr = self.load_bn_layer(curr)?;
        let curr = self.load_activation(curr)?;
        let curr = self.load_conv_layer(curr)?;

        let result = self.graph.add(curr, input);
        self.graph.set_debug_id(result, name);
        Ok(result)
    }

    fn load_gpool_block(&mut self, input: Value) -> Result<Value> {
        let name = self.read_str()?;
        println!("Parsing '{}' as gpool_block", name);

        let curr = input;
        let curr = self.load_bn_layer(curr)?;
        let curr = self.load_activation(curr)?;

        let regular_conv = self.load_conv_layer(curr)?;

        let gpool_conv = self.load_conv_layer(curr)?;
        let gpool_bn = self.load_bn_layer(gpool_conv)?;
        let gpool_act = self.load_activation(gpool_bn)?;
        let pooled = self.kata_gpool(gpool_act, PoolKind::KataGPool);
        let gpool_to_bias_mul = self.load_matmul_layer(pooled)?;

        let unsqueeze_shape = self.graph[gpool_to_bias_mul].shape.clone().concat(&shape![1, 1]);
        let bias = self.graph.view(gpool_to_bias_mul, unsqueeze_shape);

        let curr = self.graph.add(regular_conv, bias);

        let curr = self.load_bn_layer(curr)?;
        let curr = self.load_activation(curr)?;
        let curr = self.load_conv_layer(curr)?;

        let result = self.graph.add(curr, input);
        self.graph.set_debug_id(result, name);
        Ok(result)
    }

    fn kata_gpool(&mut self, input: Value, kind: PoolKind) -> Value {
        let mask_sum = (self.board_size * self.board_size) as f32;
        let mask_sum_sqrt_offset = mask_sum.sqrt() - 14.0;
        let scale_size = mask_sum_sqrt_offset / 10.0;
        let scale_area = mask_sum_sqrt_offset * mask_sum_sqrt_offset / 100.0 - 0.1;

        let mean = self.graph.reduce(input, vec![2, 3], ReduceOp::Mean);
        let max = self.graph.reduce(input, vec![2, 3], ReduceOp::Max);

        let scale_size = self.graph.scalar(scale_size);
        let mean_size = self.graph.mul(mean, scale_size);

        let values = match kind {
            PoolKind::KataGPool => vec![mean, mean_size, max],
            PoolKind::KataValueHeadGPool => {
                let scale_area = self.graph.scalar(scale_area);
                let mean_area = self.graph.mul(mean, scale_area);
                vec![mean, mean_size, mean_area]
            }
        };

        let result = self.graph.concat(values, 1, None);
        result
    }

    fn load_conv_layer(&mut self, input: Value) -> Result<Value> {
        let name = self.read_str()?;
        println!("Parsing '{}' as conv", name);

        let conv_y_size: usize = self.read_int()?;
        let conv_x_size: usize = self.read_int()?;
        let in_channels: usize = self.read_int()?;
        let out_channels: usize = self.read_int()?;
        let dilation_y: usize = self.read_int()?;
        let dilation_x: usize = self.read_int()?;

        let weights_transpose = self.read_floats(&[conv_y_size, conv_x_size, in_channels, out_channels])?;
        let weights = weights_transpose.permuted_axes(IxDyn(&[3, 2, 0, 1]));
        let filter = self.graph.constant_tensor(weights);

        assert!(dilation_y == 1 && dilation_x == 1);
        assert!(conv_y_size % 2 == 1 && conv_x_size % 2 == 1);
        let result = self.graph.conv(input, filter, 1, 1, conv_y_size / 2, conv_x_size / 2);

        self.graph.set_debug_id(result, name);
        Ok(result)
    }

    fn load_matmul_layer(&mut self, input: Value) -> Result<Value> {
        let name = self.read_str()?;
        println!("Parsing '{}' as matmul", name);

        let in_channels: usize = self.read_int()?;
        let out_channels: usize = self.read_int()?;

        let weight = self.read_floats(&[in_channels, out_channels])?;
        let weight = self.graph.constant_tensor(weight);
        let result = self.graph.mat_mul(input, weight);

        self.graph.set_debug_id(result, name);
        Ok(result)
    }

    fn load_mat_bias_layer(&mut self, input: Value) -> Result<Value> {
        let name = self.read_str()?;
        let channels: usize = self.read_int()?;

        let weight = self.read_floats(&[channels])?;
        let weight = self.graph.constant_tensor(weight);
        let result = self.graph.add(input, weight);

        self.graph.set_debug_id(result, name);
        Ok(result)
    }

    fn load_bn_layer(&mut self, input: Value) -> Result<Value> {
        let name = self.read_str()?;
        println!("Parsing '{}' as bn_layer", name);

        let channels: usize = self.read_int()?;
        let epsilon: f32 = self.read_float()?;
        let has_scale = self.read_bool()?;
        let has_bias = self.read_bool()?;

        // mean/variance normalization
        println!("reading mean");
        let mean = self.read_floats(&[1, channels, 1, 1])?;
        let mean = self.graph.constant_tensor(mean);

        println!("reading variance");
        let variance = self.read_floats(&[1, channels, 1, 1])?;
        let variance = self.graph.constant_tensor(variance);

        let curr = self.graph.sub(input, mean);
        let const_eps = self.graph.scalar(epsilon);
        let div_squared = self.graph.add(variance, const_eps);
        let div = self.graph.unary(UnaryOp::Sqrt, div_squared);
        let curr = self.graph.binary(BinaryOp::Div, curr, div);

        // scale
        let curr = if has_scale {
            println!("reading scale");
            let scale = self.read_floats(&[1, channels, 1, 1])?;
            let scale = self.graph.constant_tensor(scale);

            self.graph.mul(curr, scale)
        } else {
            println!("skipping scale");
            curr
        };

        // bias
        assert!(has_bias);
        println!("reading bias");
        let bias = self.read_floats(&[1, channels, 1, 1])?;
        let bias = self.graph.constant_tensor(bias);
        let curr = self.graph.add(curr, bias);

        self.graph.set_debug_id(curr, name);
        Ok(curr)
    }

    fn load_activation(&mut self, input: Value) -> Result<Value> {
        let name = self.read_str()?;

        let result = if self.version >= 11 {
            let kind = self.read_str()?;
            println!("Parsing '{}' as activation, kind '{}'", name, kind);

            match &*kind {
                "ACTIVATION_IDENTITY" => input,
                "ACTIVATION_RELU" => self.graph.relu(input),
                "ACTIVATION_MISH" => self.graph.unary(UnaryOp::Mish, input),
                _ => return Err(Error::UnsupportedActivationKind(kind)),
            }
        } else {
            println!("Parsing '{}' as activation", name);
            self.graph.relu(input)
        };

        self.graph.set_debug_id(result, name);
        Ok(result)
    }

    fn read_str(&mut self) -> Result<String> {
        let mut result = String::new();

        loop {
            self.input.read_line(&mut result)?;

            if result.len() == 0 {
                return Err(Error::Eof);
            }
            if result.trim().len() > 0 {
                return Ok(result.trim().to_owned());
            }
        }
    }

    fn read_int<T: FromStr<Err = ParseIntError>>(&mut self) -> Result<T> {
        let line = self.read_str()?;
        Ok(line.parse().map_err(|e| Error::ParseInt(line, e)).unwrap())
    }

    fn read_float<T: FromStr<Err = ParseFloatError>>(&mut self) -> Result<T> {
        let line = self.read_str()?;
        Ok(line.parse().map_err(|e| Error::ParseFloat(line, e)).unwrap())
    }

    fn read_bool(&mut self) -> Result<bool> {
        let line = self.read_str()?;
        match &*line {
            "1" => Ok(true),
            "0" => Ok(false),
            _ => Err(Error::ParseBool(line)),
        }
    }

    fn read_floats(&mut self, shape: &[usize]) -> Result<Tensor> {
        const BIN_PREFIX: &[u8] = b"@BIN@";
        let mut check = [0; BIN_PREFIX.len()];
        self.input.read_exact(&mut check)?;
        if check != BIN_PREFIX {
            return Err(Error::InvalidBinPrefix(String::from_utf8_lossy(&check).into_owned()));
        }

        let mut data = vec![0.0; shape.iter().copied().product()];
        self.input.read_f32_into::<LittleEndian>(&mut data)?;

        // drop trailing newline
        let mut s = String::new();
        self.input.read_line(&mut s)?;
        assert!(s.trim().is_empty());

        Ok(Tensor::from_shape_vec(IxDyn(shape), data).unwrap())
    }
}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Error::IO(value)
    }
}
