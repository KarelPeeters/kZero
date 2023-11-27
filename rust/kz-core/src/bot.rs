use std::fmt::Debug;

use async_trait::async_trait;
use board_game::ai::Bot;
use board_game::board::Board;
use board_game::board::BoardDone;
use decorum::N32;
use internal_iterator::InternalIterator;
use rand::Rng;

use kz_util::sequence::choose_max_by_key;

use crate::network::EvalClient;
use crate::zero::step::QMode;

#[async_trait]
pub trait AsyncBot<B: Board> {
    async fn select_move(&mut self, board: &B) -> Result<B::Move, BoardDone>;
}

#[derive(Debug)]
pub struct WrapAsync<T>(pub T);

#[async_trait]
impl<B: Board, T: Bot<B> + Send> AsyncBot<B> for WrapAsync<T> {
    async fn select_move(&mut self, board: &B) -> Result<B::Move, BoardDone> {
        self.0.select_move(board)
    }
}

#[derive(Debug)]
pub struct MaxValueBot<B, R> {
    pub eval_client: EvalClient<B>,
    pub rng: R,
    pub q_mode: QMode,
}

#[derive(Debug)]
pub struct MaxPolicyBot<B, R> {
    pub eval_client: EvalClient<B>,
    pub rng: R,
}

#[async_trait]
impl<B: Board, R: Rng + Send> AsyncBot<B> for MaxValueBot<B, R> {
    async fn select_move(&mut self, board: &B) -> Result<B::Move, BoardDone> {
        let boards: Vec<B> = board.children()?.map(|(_, child)| child).collect();
        let evals = self.eval_client.map_async(boards).await;

        let index = choose_max_by_key(
            0..evals.len(),
            |&index| {
                let values = evals[index].values.parent_flip();
                let value = self.q_mode.select(values.value, values.wdl).value;
                N32::from(value)
            },
            &mut self.rng,
        )
        .unwrap();

        let mv = board.available_moves().unwrap().nth(index).unwrap();
        Ok(mv)
    }
}

#[async_trait]
impl<B: Board, R: Rng + Send> AsyncBot<B> for MaxPolicyBot<B, R> {
    async fn select_move(&mut self, board: &B) -> Result<B::Move, BoardDone> {
        board.check_done()?;

        let eval = self.eval_client.map_async_single(board.clone()).await;
        let index = choose_max_by_key(
            0..eval.policy.len(),
            |&index| N32::from(eval.policy[index]),
            &mut self.rng,
        )
        .unwrap();

        let mv = board.available_moves().unwrap().nth(index).unwrap();
        Ok(mv)
    }
}
