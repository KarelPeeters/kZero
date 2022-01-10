import itertools
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as nnf
from torch.optim import AdamW
from torchvision.datasets import MNIST, CIFAR10

from lib.logger import Logger
from lib.plotter import run_with_plotter, LogPlotter
from quantized.binary import BLinear, BSign
from quantized.quant_bits import QLinear

DEVICE = "cpu"


def build_network(i_s: int, i_c: int, o_c: int):
    return nn.Sequential(
        nn.Flatten(),
        BSign(),
        BLinear(i_c * i_s * i_s, 256),
        BSign(),
        BLinear(256, o_c)
    )


def eval(network, x, y_target):
    y = network(x)
    loss = nnf.cross_entropy(y, y_target)
    acc = (torch.argmax(y, dim=1) == y_target).sum() / len(y_target)
    return loss, acc


def dataset_to_tensors(dataset):
    if isinstance(dataset, MNIST):
        x = dataset.data.unsqueeze(1).float() / 255
        y = dataset.targets
    elif isinstance(dataset, CIFAR10):
        x = torch.tensor(np.moveaxis(dataset.data, 3, 1), dtype=torch.float) / 255
        y = torch.tensor(dataset.targets, dtype=torch.int64)
    else:
        assert False, f"Unknown dataset type '{dataset}'"

    return x.to(DEVICE), y.to(DEVICE)


def sample(data, batch_size):
    x, y = data
    assert len(x) == len(y)
    i = torch.randint(len(x), (batch_size,))
    return x[i], y[i]


def log_param_scale(logger: Logger, name: str, param):
    logger.log("scale", f"{name} min", param.min())
    logger.log("scale", f"{name} mean", param.mean())
    logger.log("scale", f"{name} max", param.max())


def main(plotter: Optional[LogPlotter]):
    train_data = dataset_to_tensors(MNIST("../ignored/data", train=True, download=True))
    test_data = dataset_to_tensors(MNIST("../ignored/data", train=False, download=True))

    network = build_network(28, 1, 10)
    network.to(DEVICE)

    # opt = SGD(network.parameters(), lr=0.001, momentum=0.09, weight_decay=1e-2)
    opt = AdamW(network.parameters(), weight_decay=1e-1)
    batch_size = 256

    logger = Logger()

    for _ in itertools.count():
        if plotter:
            plotter.block_while_paused()

        logger.start_batch()

        network.eval()
        test_x, test_y_target = sample(train_data, batch_size)
        test_loss, test_acc = eval(network, test_x, test_y_target)

        network.train()
        train_x, train_y_target = sample(test_data, batch_size)
        train_loss, train_acc = eval(network, train_x, train_y_target)

        opt.zero_grad(set_to_none=True)
        train_loss.backward()
        opt.step()

        logger.log("acc", "test", test_acc.item())
        logger.log("acc", "train", train_acc.item())
        logger.log("loss", "test", test_loss.item())
        logger.log("loss", "train", train_loss.item())

        for (mi, module) in enumerate(network.modules()):
            if isinstance(module, BLinear):
                log_param_scale(logger, f"{mi} w", module.weight)
            if isinstance(module, QLinear):
                log_param_scale(logger, f"{mi} w", module.linear.weight)
                if module.linear.bias is not None:
                    log_param_scale(logger, f"{mi} b", module.linear.bias)
            if isinstance(module, nn.Linear):
                log_param_scale(logger, f"{mi} w", module.weight)
                if module.bias is not None:
                    log_param_scale(logger, f"{mi} b", module.bias)

        if plotter:
            plotter.update(logger)


if __name__ == '__main__':
    run_with_plotter(main)
