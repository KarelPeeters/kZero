use kn_cuda_eval::executor::CudaExecutor;
use kn_cuda_sys::wrapper::handle::Device;
use kn_graph::dtype::{DTensor, DType, Tensor};
use kn_graph::graph::Graph;
use kn_graph::shape;
use kz_util::throughput::PrintThroughput;

fn main() {
    unsafe {
        main_inner();
    }
}

unsafe fn main_inner() {
    let size = 256 * 1024 * 1024;

    let graph = {
        let mut graph = Graph::new();

        let shape = shape![size];
        let input = graph.input(shape, DType::F32);

        let result = (0..10).fold(input, |curr, _| graph.add(curr, curr));

        graph.output(result);

        graph
    };

    let device = Device::new(0);

    println!("{:#?}", device.properties());

    let threads = 2;

    let (sender, receiver) = flume::bounded(threads);

    crossbeam::scope(|s| {
        for _ in 0..threads {
            s.spawn(|_| {
                let input_buffer = DTensor::F32(Tensor::zeros(vec![size]));

                let mut exec = CudaExecutor::new(device, &graph, 0);

                loop {
                    exec.evaluate(&[input_buffer.clone()]);
                    // exec.run_async();
                    // exec.stream().synchronize();

                    if sender.send(()).is_err() {
                        break;
                    }
                }
            });
        }

        s.spawn(move |_| {
            let mut tp = PrintThroughput::new("evals");
            for () in &receiver {
                tp.update_delta(1);

                if tp.total_count() >= 10 {
                    break;
                }
            }
            drop(receiver);
        });
    })
    .unwrap();
}
