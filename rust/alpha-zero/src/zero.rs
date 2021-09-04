use std::convert::TryInto;
use std::fmt::{Debug, Display, Formatter};
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::ops::{Index, IndexMut};

use decorum::N32;
use internal_iterator::InternalIterator;
use itertools::{Itertools, zip_eq};
use rand::{Rng};
use rand::prelude::IteratorRandom;
use rand_distr::Distribution;
use unwrap_match::unwrap_match;

use board_game::ai::Bot;
use board_game::board::{Board, Outcome};
use board_game::symmetry::{Symmetry, SymmetryDistribution};
use board_game::wdl::{Flip, OutcomeWDL, POV, WDL};

use crate::network::{Network, ZeroEvaluation};

#[derive(Debug, Copy, Clone)]
pub struct ZeroSettings {
    pub batch_size: usize,
    pub exploration_weight: f32,
    pub random_symmetries: bool,
}

impl ZeroSettings {
    pub fn new(batch_size: usize, exploration_weight: f32, random_symmetries: bool) -> Self {
        ZeroSettings { batch_size, exploration_weight, random_symmetries }
    }

    fn gen_symmetry<S: Symmetry>(&self, rng: &mut impl Rng) -> S {
        if self.random_symmetries {
            SymmetryDistribution.sample(rng)
        } else {
            S::identity()
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct IdxRange {
    pub start: NonZeroUsize,
    pub length: u8,
}

impl IdxRange {
    pub fn new(start: usize, end: usize) -> IdxRange {
        assert!(end > start, "Cannot have empty children");
        IdxRange {
            start: NonZeroUsize::new(start).expect("start cannot be 0"),
            length: (end - start).try_into().expect("length doesn't fit"),
        }
    }

    pub fn iter(&self) -> std::ops::Range<usize> {
        self.start.get()..(self.start.get() + self.length as usize)
    }

    pub fn get(&self, index: usize) -> usize {
        assert!(index < self.length as usize);
        self.start.get() + index
    }
}

impl IntoIterator for IdxRange {
    type Item = usize;
    type IntoIter = std::ops::Range<usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

//TODO look into Node struct size

#[derive(Debug, Clone)]
pub struct Node<M> {
    pub parent: usize,
    pub last_move: Option<M>,
    pub children: Option<IdxRange>,

    /// The number of times this node (or its children) has been visited.
    /// For unsolved nodes it should always match `total_wdl().sum()`.
    pub full_visits: u64,

    /// The wdl returned by the network for this position, from the POV of the next player.
    pub net_wdl: Option<WDL<f32>>,
    /// The prior policy as evaluated by the network when the parent node was expanded.
    pub net_policy: f32,

    /// The wdl for this node from the POV of the next player.
    pub kind: NodeKind,

    /// The current virtual loss applied to this node. These visits are not counted in `full_visits` yet.
    pub virtual_loss: i32,
}

#[derive(Debug, Copy, Clone)]
pub enum NodeKind {
    Estimate { total_wdl: WDL<f32> },
    Solved { outcome: OutcomeWDL },
}

impl<N> Node<N> {
    fn new(parent: usize, last_move: Option<N>, outcome: Option<OutcomeWDL>, net_policy: f32) -> Self {
        let kind = match outcome {
            None => NodeKind::Estimate { total_wdl: WDL::default() },
            Some(outcome) => NodeKind::Solved { outcome },
        };

        Node {
            parent,
            last_move,
            children: None,

            full_visits: 0,

            net_wdl: None,
            net_policy,
            kind,
            virtual_loss: 0,
        }
    }

    pub fn add_outcome(&mut self, outcome: OutcomeWDL) {
        self.full_visits += 1;
        let total_wdl = unwrap_match!(&mut self.kind, NodeKind::Estimate { total_wdl } => total_wdl);
        *total_wdl += outcome.to_wdl();
    }

    pub fn total_visits(&self) -> u64 {
        return self.full_visits + (self.virtual_loss as u64);
    }

    pub fn is_unvisited(&self) -> bool {
        self.total_visits() == 0 && matches!(self.kind, NodeKind::Estimate { .. })
    }

    /// The (normalized) WDL of this node from the POV of the next player.
    /// Does not include virtual loss. Returns NaN if this node if unsolved and has zero total_wdl.
    pub fn wdl(&self) -> WDL<f32> {
        match self.kind {
            NodeKind::Estimate { total_wdl } => total_wdl / total_wdl.sum(),
            NodeKind::Solved { outcome } => outcome.to_wdl(),
        }
    }

    /// The solution of this node from the POV of the next player.
    /// None if this node it not (yet) solved.
    pub fn solution(&self) -> Option<OutcomeWDL> {
        match self.kind {
            NodeKind::Estimate { .. } => None,
            NodeKind::Solved { outcome } => Some(outcome),
        }
    }

    /// The UCT value of this node, including virtual loss. Returns NaN if this node has never been visited.
    pub fn uct(&self, exploration_weight: f32, parent_total_visits: u64) -> f32 {
        // things are negated since we care about the previous player POV here
        match self.kind {
            NodeKind::Estimate { total_wdl } => {
                let q = if self.total_visits() == 0 {
                    0.0
                } else {
                    //TODO add virtual loss back
                    (-total_wdl.value() - ((self.virtual_loss) as f32)) / ((self.total_visits()) as f32)
                };
                //TODO this should be -1 probably, but then we don't visit the max-policy child anymore!
                let u = self.net_policy * ((parent_total_visits) as f32).sqrt() / (1 + self.total_visits()) as f32;
                q + exploration_weight * u
            }
            NodeKind::Solved { outcome } => {
                //don't take exploration into account here, we don't want to explore this node
                //TODO what's the justification for this? It does fix an uct bug...
                // let u = self.net_policy * ((parent_total_visits) as f32).sqrt() / (1 + self.full_visits) as f32;
                -outcome.inf_sign::<f32>()
            }
        }
    }
}

/// A small wrapper type for Vec<Node> that uses u64 for indexing instead.
#[derive(Debug, Clone)]
pub struct Tree<B: Board> {
    root_board: B,
    nodes: Vec<Node<B::Move>>,
}

impl<B: Board> Index<usize> for Tree<B> {
    type Output = Node<B::Move>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.nodes[index]
    }
}

impl<B: Board> IndexMut<usize> for Tree<B> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.nodes[index]
    }
}

#[derive(Debug)]
pub enum KeepResult<B: Board> {
    Done(Outcome),
    Tree(Tree<B>),
}

impl<B: Board> Tree<B> {
    pub fn new(root_board: B) -> Self {
        assert!(!root_board.is_done(), "Cannot build tree for done board");

        let root_outcome = root_board.outcome().pov(root_board.next_player());
        let root = Node::new(0, None, root_outcome, f32::NAN);

        Tree { root_board, nodes: vec![root] }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn root_board(&self) -> &B {
        &self.root_board
    }

    pub fn best_child_of(&self, node: usize, rng: &mut impl Rng) -> usize {
        let node = &self[node];
        let children = node.children
            .expect("Root node must have expanded children");

        match node.solution() {
            Some(best_outcome) => {
                // this means the root node was solved, and we can pick a random move with this proven outcome
                let child = children.iter()
                    .filter(|&c| self[c].solution() == Some(best_outcome.flip()))
                    .choose(rng)
                    .unwrap();

                child
            }
            None => {
                // pick the most visited non-lost child
                // if all children are lost the root node would have been solved, so we're guaranteed to find one
                //TODO why is a lost child so often the most visited one?
                children.iter()
                    .filter(|&c| self[c].solution() != Some(OutcomeWDL::Win) && !self[c].is_unvisited())
                    .max_by_key(|&c| self[c].full_visits)
                    .unwrap()
            }
        }
    }

    pub fn best_move(&self, rng: &mut impl Rng) -> B::Move {
        let best_child = self.best_child_of(0, rng);
        self[best_child].last_move.unwrap()
    }

    /// The WDL of `root_board` from the POfV of `root_board.next_player`.
    pub fn wdl(&self) -> WDL<f32> {
        self[0].wdl()
    }

    /// Return the (normalized) policy vector for the root node.
    pub fn policy(&self) -> impl Iterator<Item=f32> + '_ {
        //TODO incorporate solved moves here
        assert!(false);

        // TODO more generally, if the root node is solved set the policy uniformly to optimal moves
        //   otherwise, set losses to zero (what about draws?)
        //   maybe set policy to whetever the visit count should have been to get get the matching uct?

        // assert!(self.len() > 1, "Must have run for at least 1 iteration");

        self[0].children.unwrap().iter().map(move |c| {
            (self[c].full_visits as f32) / ((self[0].full_visits - 1) as f32)
        })
    }

    /// Return a new tree containing the nodes that are still relevant after playing the given move.
    /// Effectively this copies the part of the tree starting from the selected child.
    pub fn keep_move(&self, _: B::Move) -> KeepResult<B> {
        todo!("re-implement this");
    }

    #[must_use]
    pub fn display(&self, max_depth: usize, full: bool) -> TreeDisplay<B> {
        let parent_visits = self[0].full_visits;
        TreeDisplay { tree: self, node: 0, curr_depth: 0, max_depth, parent_visits, full }
    }
}

#[derive(Debug)]
pub struct TreeDisplay<'a, B: Board> {
    tree: &'a Tree<B>,
    node: usize,
    curr_depth: usize,
    max_depth: usize,
    parent_visits: u64,
    full: bool,
}

impl<B: Board> Display for TreeDisplay<'_, B> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        //TODO use some fixed source of randomness here

        if self.curr_depth == 0 {
            // writeln!(f, "best_move: {:?}", self.tree.best_move(&mut thread_rng()))?;
            writeln!(f, "move: visits virt uct zero(w/d/l, policy) net(w/d/l, policy)")?;
        }

        for _ in 0..self.curr_depth { write!(f, "  ")? }

        let node = &self.tree[self.node];
        let tmp;
        let zero_wdl = match node.kind {
            NodeKind::Estimate { total_wdl } => {
                let wdl = total_wdl / total_wdl.sum();
                tmp = format!("{:.3}|{:.3}|{:.3}", wdl.win, wdl.draw, wdl.loss);
                &*tmp
            }
            NodeKind::Solved { outcome } => {
                match outcome {
                    OutcomeWDL::Win => "W",
                    OutcomeWDL::Draw => "D",
                    OutcomeWDL::Loss => "L",
                }
            }
        };

        let net_wdl = node.net_wdl.unwrap_or(WDL::nan());

        //TODO don't hardcode exploration weight here
        writeln!(
            f,
            "{:?}: {} {} {:.4} zero({}, {:.4}) net({:.3}|{:.3}|{:.3}, {:.4})",
            node.last_move, node.full_visits, -node.virtual_loss,
            node.uct(2.0, self.tree[node.parent].total_visits()),
            zero_wdl,
            (node.full_visits as f32) / (self.parent_visits as f32),
            net_wdl.win, net_wdl.draw, net_wdl.loss,
            node.net_policy,
        )?;

        if self.curr_depth == self.max_depth { return Ok(()); }

        if let Some(children) = node.children {
            //let best_child = self.tree.best_child_of(self.node, &mut thread_rng());
            let best_child = 0;

            for child in children {
                let next_max_depth = if self.full || child == best_child {
                    self.max_depth
                } else {
                    self.curr_depth + 1
                };

                let child_display = TreeDisplay {
                    tree: self.tree,
                    node: child,
                    curr_depth: self.curr_depth + 1,
                    max_depth: next_max_depth,
                    parent_visits: node.full_visits,
                    full: self.full,
                };
                write!(f, "{}", child_display)?;
            }
        }

        Ok(())
    }
}

/// A coroutine-style implementation that yields `Request`s instead of immediately calling a network.
#[derive(Debug, Clone)]
pub struct ZeroState<B: Board> {
    pub tree: Tree<B>,
    pub target_iterations: u64,
    settings: ZeroSettings,
    expected_nodes: Vec<usize>,
}

#[derive(Debug)]
pub enum RunResult<B: Board> {
    Request(BatchRequest<B>),
    Done,
}

#[derive(Debug)]
pub struct BatchRequest<B: Board> {
    subs: Vec<Request<B>>,
}

#[derive(Debug)]
pub struct BatchResponse<B: Board> {
    subs: Vec<Response<B>>,
}

#[derive(Debug, Clone)]
pub struct Request<B: Board> {
    curr_board: B,
    curr_node: usize,
    sym: B::Symmetry,
}

impl<B: Board> Request<B> {
    pub fn board(&self) -> B {
        self.curr_board.map(self.sym)
    }
}

#[derive(Debug)]
pub struct Response<B: Board> {
    pub request: Request<B>,
    pub evaluation: ZeroEvaluation,
}

#[derive(Debug)]
enum StepResult<B: Board> {
    Backprop(OutcomeWDL),
    Proven(OutcomeWDL),
    Request(Request<B>),
}

impl<B: Board> Tree<B> {
    /// Run a single step of the MCTS algorithm. Keeps going until it has proven `curr_node` or a request is generated.
    ///
    /// Recursively descend down the tree, picking unexplored or else max-uct children, util:
    /// * an unvisited node is hit, in which case
    ///     * `visit_count` and `virtual_loss` are incremented for all nodes below and including `curr_node`
    ///     * the associated request is returned
    /// * a solved node is hit, in which case
    ///     * `visit_count` is incremented for all nodes below and including `curr_node`
    ///     * the outcome from the POV of `curr_board.next_player()` is returned
    ///
    /// During this process nodes are marked proven if possible based on their children.
    fn recursive_step(
        &mut self,
        curr_node: usize,
        curr_board: B,
        settings: ZeroSettings,
        rng: &mut impl Rng,
    ) -> StepResult<B> {
        // if this node is already solved return that outcome
        if let NodeKind::Solved { outcome } = self[curr_node].kind {
            //TODO check if this is ever triggered and how (probably when a board is actually solved?)
            self[curr_node].full_visits += 1;
            return StepResult::Proven(outcome);
        }

        // if this is the first visit initialize the children
        let children = match self[curr_node].children {
            None => {
                // create all child nodes
                let child_count = curr_board.available_moves().count();
                let uniform_policy = 1.0 / (child_count as f32);

                let start = self.len();
                curr_board.available_moves().for_each(|mv: B::Move| {
                    let next_board = curr_board.clone_and_play(mv);
                    let outcome = next_board.outcome().pov(next_board.next_player());

                    //initialize with uniform policy, will be replaced later
                    let node = Node::new(curr_node, Some(mv), outcome, uniform_policy);
                    self.nodes.push(node);
                });
                let end = self.len();

                let children = IdxRange::new(start, end);
                self[curr_node].children = Some(children);

                // check if we can immediately prove this node
                if let Some(outcome) = self.try_solve_node_and_inc(curr_node, children) {
                    return StepResult::Proven(outcome);
                }

                // add virtual loss to this node
                //TODO try without too and see the effect on strength
                //  err doesn't make sense, you'll almost always get the same request twice!
                self[curr_node].virtual_loss += 1;

                // return the request
                let sym = settings.gen_symmetry(rng);
                let request = Request { curr_board, curr_node, sym };

                return StepResult::Request(request);
            }
            Some(children) => children,
        };

        // pick a random unvisited child if any
        //TODO make random again
        // let picked_unvisited = unvisited.choose(rng);
        // let mut unvisited = children.iter().filter(|&c| self[c].is_unvisited());
        // let picked_unvisited = unvisited.next();
        let picked_unvisited = None;

        let next_node = if let Some(picked_unvisited) = picked_unvisited {
            picked_unvisited
        } else {
            // otherwise pick a random max-uct child
            //TODO solver change here? avoid proven loss? (and draw?)
            // there can't be a win since then this node would be proven already
            let parent_visits = self[curr_node].total_visits();
            //TODO make random again
            // let picked_child = choose_max_by_key(
            //     children.iter(),
            //     |&c| N32::from(self[c].uct(settings.exploration_weight, parent_visits)),
            //     rng,
            // ).unwrap();
            let picked_child = children.iter()
                .max_by_key(|&c| N32::from(self[c].uct(settings.exploration_weight, parent_visits)))
                .unwrap();

            picked_child
        };

        // continue recursing
        let next_mv = self[next_node].last_move.unwrap();
        let next_board = curr_board.clone_and_play(next_mv);
        let result = self.recursive_step(next_node, next_board, settings, rng);

        // handle the result
        match result {
            StepResult::Backprop(child_outcome) => {
                // continue backprop
                let curr_outcome = child_outcome.flip();
                self[curr_node].add_outcome(curr_outcome);
                StepResult::Backprop(curr_outcome)
            }
            StepResult::Proven(child_outcome) => {
                //TODO instead of re-iterating every time we can also keep state that's in OutcomeWDL::best in the parent node
                if let Some(outcome) = self.try_solve_node_and_inc(curr_node, children) {
                    // continue proving
                    StepResult::Proven(outcome)
                } else {
                    // switch to backprop
                    let curr_outcome = child_outcome.flip();
                    self[curr_node].add_outcome(curr_outcome);
                    StepResult::Backprop(curr_outcome)
                }
            }
            StepResult::Request(request) => {
                // increment the virtual loss of this path and pass the request on
                self[curr_node].virtual_loss += 1;
                StepResult::Request(request)
            }
        }
    }

    /// Check if we can solve this node based on the child node solutions,
    /// and if so increment this nodes `visit_count` and mark it as solved.
    /// Returns the outcome from the POV of `node.next_player`.
    fn try_solve_node_and_inc(&mut self, node: usize, children: IdxRange) -> Option<OutcomeWDL> {
        assert!(
            matches!(self[node].kind, NodeKind::Estimate { .. }),
            "node cannot already be proven"
        );

        let outcome = OutcomeWDL::best(children.iter().map(|c| self[c].solution().flip()));
        if let Some(outcome) = outcome {
            self[node].full_visits += 1;
            self[node].kind = NodeKind::Solved { outcome };
        }
        outcome
    }

    /// Propagate the given `wdl` up to the root, starting from `node`.
    /// The initial `wdl` if from the POV of the next player after `node`.
    /// Intermediate solved nodes are ignored without stopping propagation.
    fn propagate_wdl(&mut self, node: usize, wdl: WDL<f32>) {
        let mut curr = node;
        let mut curr_wdl = wdl;

        loop {
            let curr_node = &mut self[curr];

            assert!(curr_node.virtual_loss > 0);
            curr_node.virtual_loss -= 1;
            curr_node.full_visits += 1;

            match &mut curr_node.kind {
                NodeKind::Estimate { total_wdl } => { *total_wdl += curr_wdl }
                NodeKind::Solved { .. } => {}
            }

            if curr == 0 { break; }
            curr = self[curr].parent;
            curr_wdl = curr_wdl.flip()
        }
    }
}

impl<B: Board> ZeroState<B> {
    /// Create a new state that will expand the given tree until its root node has been visited `iterations` times.
    pub fn new(tree: Tree<B>, target_iterations: u64, settings: ZeroSettings) -> ZeroState<B> {
        Self { tree, target_iterations, settings, expected_nodes: vec![] }
    }

    /// Run until finished or a network evaluation is needed.
    pub fn run_until_result(
        &mut self,
        response: Option<BatchResponse<B>>,
        rng: &mut impl Rng,
        stop_cond: &mut impl FnMut() -> bool,
    ) -> RunResult<B> {
        //apply the previous network evaluation if any
        match response {
            None => assert!(self.expected_nodes.is_empty(), "Expected evaluation response, got nothing"),
            Some(response) => {
                assert_eq!(self.expected_nodes.len(), response.subs.len(), "Unexpected number of responses");
                self.apply_batch_response(response)
            }
        }

        //continue running
        self.gather_new_batch_request(rng, stop_cond)
    }

    /// Continue running, starting from the selection phase at the root of the tree.
    fn gather_new_batch_request(
        &mut self,
        rng: &mut impl Rng,
        stop_cond: &mut impl FnMut() -> bool,
    ) -> RunResult<B> {
        let mut sub_requests = vec![];

        //TODO it's probably more useful to count net evals instead of visits, since some prove visits are almost instant
        while self.tree[0].full_visits < self.target_iterations {
            if stop_cond() { return RunResult::Done; }
            if sub_requests.len() == self.settings.batch_size { break; }

            let root_board = self.tree.root_board.clone();
            let step_result = self.tree.recursive_step(0, root_board, self.settings, rng);

            match step_result {
                StepResult::Backprop(_) => continue,
                StepResult::Proven(_) => return RunResult::Done,
                StepResult::Request(request) => {
                    self.expected_nodes.push(request.curr_node);
                    sub_requests.push(request)
                }
            }
        }

        if sub_requests.is_empty() {
            // no requests, this means the iteration limit ran out exactly and we can stop immediately
            RunResult::Done
        } else {
            RunResult::Request(BatchRequest { subs: sub_requests })
        }
    }

    /// Insert the given network evaluation into the current tree.
    fn apply_batch_response(&mut self, response: BatchResponse<B>) {
        // temporarily take it out for the borrow checker
        let mut expected_nodes = std::mem::take(&mut self.expected_nodes);

        for (response, &expected_node) in zip_eq(response.subs, &expected_nodes) {
            // unwrap everything
            let Response { request, evaluation } = response;
            let ZeroEvaluation { wdl, policy: sym_policy } = evaluation;
            let Request { curr_board, curr_node, sym } = request;

            // is this actually our request?
            assert_eq!(expected_node, curr_node, "Received response for wrong node");

            let tree = &mut self.tree;

            // does this node already have a response?
            assert!(tree[curr_node].net_wdl.is_none(), "Node already has net_wdl, it already got a response!");
            tree[curr_node].net_wdl = Some(wdl);

            // set the policy
            for_each_original_move_and_policy(&curr_board, sym, &sym_policy, |i, _, p| {
                let child = tree[curr_node].children.unwrap().get(i);
                let child_node = &mut tree[child];
                child_node.net_policy = p;
            });

            // propagate the eval back up to the root node
            self.tree.propagate_wdl(curr_node, wdl);
        }

        // put the (now empty) vec back to keep the allocation
        expected_nodes.clear();
        self.expected_nodes = expected_nodes;
    }
}

/// Visit the available (move, policy) pairs of the given board,
/// assuming sym_policy is the policy evaluated on `board.map(sym)`.
fn for_each_original_move_and_policy<B: Board>(
    board: &B,
    sym: B::Symmetry,
    sym_policy: &Vec<f32>,
    mut f: impl FnMut(usize, B::Move, f32) -> (),
) {
    assert_eq!(board.available_moves().count(), sym_policy.len());

    let policy_sum: f32 = sym_policy.iter().sum();
    assert!((1.0 - policy_sum).abs() < 0.001, "Policy sum was {} != 1.0 for board {}", policy_sum, board);

    //this reverse mapping is kind of ugly but it's probably the best we can do without more constraints on
    // moves and their ordering
    let sym_moves: Vec<B::Move> = board.map(sym).available_moves().collect();

    board.available_moves().enumerate().for_each(|(i, mv)| {
        let mv: B::Move = mv;

        let sym_mv = B::map_move(sym, mv);
        let index = sym_moves.iter().position(|&cand| cand == sym_mv).unwrap();
        f(i, mv, sym_policy[index])
    });
}

/// Build a new evaluation tree search from scratch for the given `board`.
pub fn zero_build_tree<B: Board>(
    board: &B,
    iterations: u64,
    settings: ZeroSettings,
    network: &mut impl Network<B>,
    rng: &mut impl Rng,
    mut stop_cond: impl FnMut() -> bool,
) -> Tree<B> {
    let mut state = ZeroState::new(Tree::new(board.clone()), iterations, settings);

    let mut response = None;

    //TODO think about where we actually need stop_cond
    loop {
        if stop_cond() { break; }

        let result = state.run_until_result(response, rng, &mut stop_cond);
        // println!("{}", state.tree.display(1, true));

        if stop_cond() { break; }

        match result {
            RunResult::Done => break,
            RunResult::Request(request) => {
                let boards = request.subs.iter().map(|s| s.board()).collect_vec();
                let responses = network.evaluate_batch(&boards);

                if stop_cond() { break; }

                let subs = zip_eq(request.subs, responses)
                    .map(|(req, res)| Response { request: req, evaluation: res })
                    .collect_vec();
                response = Some(BatchResponse { subs })
            }
        }
    }

    return state.tree;
}

pub struct ZeroBot<B: Board, N: Network<B>, R: Rng> {
    iterations: u64,
    settings: ZeroSettings,
    network: N,
    rng: R,
    ph: PhantomData<*const B>,
}

impl<B: Board, N: Network<B>, R: Rng> Debug for ZeroBot<B, N, R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "ZeroBot {{ iterations: {:?}, settings: {:?}, network: {:?} }}", self.iterations, self.settings, self.network)
    }
}

impl<B: Board, N: Network<B>, R: Rng> ZeroBot<B, N, R> {
    pub fn new(iterations: u64, settings: ZeroSettings, network: N, rng: R) -> Self {
        ZeroBot { iterations, settings, network, rng, ph: PhantomData }
    }

    /// Utility function that builds a tree with the settings of this bot.
    pub fn build_tree(&mut self, board: &B) -> Tree<B> {
        zero_build_tree(board, self.iterations, self.settings, &mut self.network, &mut self.rng, || false)
    }
}

impl<B: Board, N: Network<B>, R: Rng> Bot<B> for ZeroBot<B, N, R> {
    fn select_move(&mut self, board: &B) -> B::Move {
        self.build_tree(board).best_move(&mut self.rng)
    }
}