use std::cmp::min;
use std::collections::VecDeque;
use std::fmt::{Debug, Formatter};

use board_game::board::Board;
use flume::{Receiver, RecvError, Selector, Sender, TryRecvError};
use futures::never::Never;
use itertools::Itertools;
use kn_graph::graph::Graph;
use kn_runtime::Device;
use superluminal_perf::{begin_event_with_color, end_event};

use kz_core::mapping::BoardMapper;
use kz_core::network::prepared::PreparedNetwork;

#[cfg(feature = "cuda")]
use kn_cuda_eval::CudaNetwork;
#[cfg(feature = "cuda")]
use kn_cuda_sys::wrapper::handle::CudaDevice;

use kz_core::network::job_channel::{Job, JobServer};
use kz_core::network::{Network, ZeroEvaluation};

use crate::server::server::InferenceDevice;
use crate::superluminal::{CL_BLUE, CL_GREEN, CL_YELLOW};

#[derive(Debug, Copy, Clone)]
pub enum RunCondition {
    FullBatch,
    JobCount(usize),
    Any,
}

pub fn batched_executor_loop<G, N, X, Y>(
    max_batch_size: usize,
    run_condition: RunCondition,
    graph_receiver: Receiver<Option<G>>,
    server: JobServer<X, Y>,
    mut load_network: impl FnMut(G) -> N,
    mut evaluate_batch: impl FnMut(&mut N, &[X]) -> Vec<Y>,
) {
    let thread_name = std::thread::current().name().unwrap_or("unnamed").to_owned();

    assert_ne!(max_batch_size, 0, "Got batch size 0 for {}", thread_name);

    let job_receiver = server.into_receiver();

    let mut state = State::new();
    let mut network: Option<N> = None;

    let mut graph_disconnected = false;

    loop {
        assert!(network.is_some() || !graph_disconnected);

        let mut selector = Selector::new();

        if !graph_disconnected {
            selector = selector.recv(&graph_receiver, Message::Graph);
        }

        if network.is_some() {
            selector = selector.recv(&job_receiver, Message::Job);
        }

        begin_event_with_color("wait", CL_YELLOW);
        let message = selector.wait();
        end_event();

        let _: Never = match message {
            Message::Graph(Ok(graph)) => {
                handle_new_graph(&mut network, graph, &mut load_network, &thread_name);
                continue;
            }

            Message::Job(job) => {
                let network = network.as_mut().unwrap();

                match job {
                    Ok(job) => {
                        state.push_job(job);

                        while state.x.len() < max_batch_size {
                            match job_receiver.try_recv() {
                                Ok(job) => state.push_job(job),
                                Err(TryRecvError::Empty) => break,
                                Err(TryRecvError::Disconnected) => continue,
                            }
                        }

                        if state.should_eval(run_condition, max_batch_size) {
                            run_eval(&mut state, network, &mut evaluate_batch, max_batch_size);
                        }

                        continue;
                    }

                    Err(RecvError::Disconnected) => {
                        assert!(state.items_to_eval() < max_batch_size);

                        if state.items_to_eval() > 0 {
                            run_eval(&mut state, network, &mut evaluate_batch, max_batch_size);
                        }

                        assert!(state.items_to_eval() == 0 && state.items_to_send() == 0);

                        return;
                    }
                }
            }

            Message::Graph(Err(RecvError::Disconnected)) => match &network {
                Some(_) => {
                    graph_disconnected = true;
                    continue;
                }

                None => {
                    if state.items_to_eval() > 0 {
                        panic!(
                            "Executor {}: graph disconnected by we still have items pending: {:?}",
                            thread_name, state
                        );
                    }

                    match job_receiver.recv() {
                        Ok(_) => panic!("Executor {}: got new job after graph disconnection", thread_name),
                        Err(RecvError::Disconnected) => {}
                    }

                    return;
                }
            },
        };
    }
}

#[derive(Debug)]
pub enum AlphaZeroExecutorNetwork<B: Board, M: BoardMapper<B>> {
    Cpu(PreparedNetwork<B, M>),

    #[cfg(feature = "cuda")]
    Cuda(CudaNetwork<M>),
}

impl<B: Board, M: BoardMapper<B>> Network<B> for AlphaZeroExecutorNetwork<B, M> {
    fn max_batch_size(&self) -> usize {
        match self {
            Self::Cpu(network) => network.max_batch_size(),

            #[cfg(feature = "cuda")]
            Self::Cuda(network) => network.max_batch_size(),
        }
    }

    fn evaluate_batch(&mut self, boards: &[impl std::borrow::Borrow<B>]) -> Vec<ZeroEvaluation<'static>> {
        match self {
            Self::Cpu(network) => network.evaluate_batch(boards),

            #[cfg(feature = "cuda")]
            Self::Cuda(network) => network.evaluate_batch(boards),
        }
    }
}

pub fn alphazero_batched_executor_loop<B: Board, M: BoardMapper<B>>(
    max_batch_size: usize,
    device: InferenceDevice,
    mapper: M,
    run_condition: RunCondition,
    graph: Graph,
    server: JobServer<B, ZeroEvaluation<'static>>,
) {
    let (graph_sender, graph_receiver) = flume::bounded(1);
    graph_sender.send(Some(graph.clone())).unwrap();
    drop(graph_sender);

    let network = match device {
        InferenceDevice::Cpu => {
            AlphaZeroExecutorNetwork::Cpu(PreparedNetwork::new(mapper, Device::Cpu, graph, max_batch_size))
        }

        #[cfg(feature = "cuda")]
        InferenceDevice::Cuda(device) => {
            AlphaZeroExecutorNetwork::Cuda(CudaNetwork::new(mapper, &graph, max_batch_size, device))
        }
    };

    let mut network = Some(network);

    batched_executor_loop(
        max_batch_size,
        run_condition,
        graph_receiver,
        server,
        move |_| network.take().unwrap(),
        |network, batch_x| network.evaluate_batch(batch_x),
    )
}

#[derive(Debug)]
enum Message<G, J> {
    Graph(G),
    Job(J),
}

struct State<X, Y> {
    x: VecDeque<X>,
    senders: VecDeque<(usize, Sender<Vec<Y>>)>,
    leftover_y: VecDeque<Y>,
}

impl<X, Y> Debug for State<X, Y> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("State")
            .field("items_to_eval", &self.items_to_eval())
            .field("items_waiting_for_send", &self.items_waiting_for_send())
            .field("items_to_send", &self.items_to_send())
            .field("senders", &self.senders.len())
            .finish()
    }
}

impl<X, Y> State<X, Y> {
    pub fn new() -> Self {
        Self {
            x: VecDeque::new(),
            senders: VecDeque::new(),
            leftover_y: VecDeque::new(),
        }
    }

    fn check_invariants(&self) {
        assert!(!self.can_fill_next_sender());
        assert_eq!(
            self.items_to_eval() + self.items_waiting_for_send(),
            self.items_to_send()
        );
    }

    fn items_to_eval(&self) -> usize {
        self.x.len()
    }

    fn items_waiting_for_send(&self) -> usize {
        self.leftover_y.len()
    }

    fn items_to_send(&self) -> usize {
        self.senders.iter().map(|&(len, _)| len).sum::<usize>()
    }

    fn push_job(&mut self, job: Job<X, Y>) {
        let Job { x, sender } = job;

        if x.len() == 0 {
            let _ = sender.send(vec![]);
        } else {
            self.senders.push_back((x.len(), sender));
            self.x.extend(x.into_iter());
        }
    }

    fn can_fill_next_sender(&self) -> bool {
        if let Some(&(len, _)) = self.senders.front() {
            self.leftover_y.len() >= len
        } else {
            false
        }
    }

    fn should_eval(&self, run_condition: RunCondition, max_batch_size: usize) -> bool {
        match run_condition {
            RunCondition::FullBatch => self.x.len() >= max_batch_size,
            RunCondition::JobCount(job_count) => self.x.len() >= job_count * max_batch_size,
            RunCondition::Any => !self.x.is_empty(),
        }
    }
}

fn run_eval<X, Y, N>(
    state: &mut State<X, Y>,
    network: &mut N,
    evaluate_batch: &mut impl FnMut(&mut N, &[X]) -> Vec<Y>,
    max_batch_size: usize,
) {
    begin_event_with_color("evaluate", CL_BLUE);

    let eval_count = min(state.x.len(), max_batch_size);
    let x = state.x.drain(..eval_count).collect_vec();
    let y = evaluate_batch(network, &x);

    end_event();

    state.leftover_y.extend(y);

    while state.can_fill_next_sender() {
        let (len, sender) = state.senders.pop_front().unwrap();
        let y = state.leftover_y.drain(..len).collect_vec();

        begin_event_with_color("send results", CL_GREEN);
        sender.send(y).unwrap();
        end_event();
    }

    state.check_invariants();
}

fn handle_new_graph<G, N>(
    network: &mut Option<N>,
    graph: Option<G>,
    load_network: &mut impl FnMut(G) -> N,
    thread_name: &str,
) {
    *network = graph.map(|graph| {
        println!("Executor {}: got new graph", thread_name);
        load_network(graph)
    });
}
