use board_game::board::Board;
use serde::{Deserialize, Serialize};

use kz_core::zero::node::UctWeights;
use kz_core::zero::step::{FpuMode, QMode};

use crate::server::serde_helper::ToFromStringArg;
use crate::simulation::Simulation;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartupSettings {
    pub game: String,
    pub muzero: bool,
    pub start_pos: String,

    pub first_gen: u32,
    pub output_folder: String,
    pub games_per_gen: usize,

    // TODO implement some kind of adaptive batch sizing, especially for root
    pub cpu_threads_per_device: usize,
    pub gpu_threads_per_device: usize,
    pub gpu_batch_size: usize,
    pub gpu_batch_size_root: usize,
    pub search_batch_size: usize,

    pub saved_state_channels: usize,
    pub eval_random_symmetries: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum Command {
    StartupSettings(StartupSettings),
    NewSettings(Settings),
    NewNetwork(String),
    WaitForNewNetwork,
    UseDummyNetwork,
    Stop,
}

#[derive(Default, Debug, Copy, Clone, Eq, PartialEq)]
pub struct Evals {
    // evals that actually happened, does not include cached
    pub real: u64,
    // evals that would have happened if the batch was full
    pub potential: u64,
    // evals that hit the cache
    pub cached: u64,
}

#[derive(Debug)]
pub enum GeneratorUpdate<B: Board> {
    Stop,

    StartedSimulation {
        generator_id: usize,
    },

    FinishedMove {
        generator_id: usize,
        curr_game_length: usize,
    },

    FinishedSimulation {
        generator_id: usize,
        simulation: Simulation<'static, B>,
    },

    ExpandEvals(Evals),
    RootEvals(Evals),
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum ServerUpdate {
    Stopped,
    FinishedFile { index: u32 },
}

//TODO split this into AlphaZero and MuZero structs, the overlap is getting pretty small
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    // self-play game affecting settings
    pub max_game_length: Option<u64>,
    pub weights: Weights,
    pub q_mode: ToFromStringArg<QMode>,

    pub temperature: f32,
    pub zero_temp_move_count: u32,

    pub dirichlet_alpha: f32,
    pub dirichlet_eps: f32,

    pub search_policy_temperature_root: f32,
    pub search_policy_temperature_child: f32,
    pub search_fpu_root: ToFromStringArg<FpuMode>,
    pub search_fpu_child: ToFromStringArg<FpuMode>,
    pub search_virtual_loss_weight: f32,

    pub full_search_prob: f64,
    pub full_iterations: u64,
    pub part_iterations: u64,

    pub top_moves: usize,

    // performance
    pub cache_size: usize,
}

#[derive(Default, Debug, Copy, Clone, Serialize, Deserialize)]
pub struct Weights {
    pub exploration_weight: Option<f32>,
    pub moves_left_weight: Option<f32>,
    pub moves_left_clip: Option<f32>,
    pub moves_left_sharpness: Option<f32>,
}

impl Weights {
    pub fn to_uct(&self) -> UctWeights {
        let default = UctWeights::default();
        UctWeights {
            exploration_weight: self.exploration_weight.unwrap_or(default.exploration_weight),
            moves_left_weight: self.moves_left_weight.unwrap_or(default.moves_left_weight),
            moves_left_clip: self.moves_left_clip.unwrap_or(default.moves_left_clip),
            moves_left_sharpness: self.moves_left_sharpness.unwrap_or(default.moves_left_sharpness),
        }
    }
}

impl Evals {
    pub fn new(real: u64, potential: u64, cached: u64) -> Self {
        Self {
            real,
            potential,
            cached,
        }
    }
}

impl std::ops::Add for Evals {
    type Output = Evals;

    fn add(self, rhs: Self) -> Self::Output {
        Evals {
            real: self.real + rhs.real,
            potential: self.potential + rhs.potential,
            cached: self.cached + rhs.cached,
        }
    }
}

impl std::ops::AddAssign for Evals {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
