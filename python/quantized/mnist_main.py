import itertools

import numpy as np
import torch
from torch import nn
from torch.nn import functional as nnf, Flatten
from torch.optim import AdamW
from torchvision.datasets import MNIST, CIFAR10
from torchvision.utils import save_image

from lib.logger import Logger
from lib.plotter import run_with_plotter
from lib.residual import ResModule
from quantized.binary import BSign, BConv2d

DEVICE = "cuda"


def FConv2d(channels_in: int, channels_out: int, filter_size: int, groups: int = 1):
    assert filter_size % 2 == 1
    padding = filter_size // 2
    return nn.Conv2d(channels_in, channels_out, (filter_size, filter_size), padding=(padding, padding), groups=groups)


def block(c: int):
    # return ResModule(
    # nn.BatchNorm2d(channels),
    # nn.ReLU(),
    # FConv2d(channels, channels, 3),
    # nn.BatchNorm2d(channels),
    # nn.ReLU(),
    # FConv2d(channels, channels, 3),
    # nn.BatchNorm2d(channels),
    # )

    module = ResModule(
        nn.BatchNorm2d(c),
        BSign(False),
        BConv2d(c, c, 3, False),
        BSign(False),
        BConv2d(c, c, 3, False),
        nn.BatchNorm2d(c)
    )
    module.inner[-1].weight.data.fill_(0.0)
    return module


def build_network(
        i_s: int, i_c: int, o_c: int, bits: int,
        clamp_sign_grad: bool, clamp_weight_grad: bool,
        dropout: float
):
    return nn.Sequential(
        FConv2d(i_c, 128, 1),
        # BFromUnitFloat(8),
        # BConv2d(8*3, 128, 1, False),

        block(128),
        block(128),
        block(128),
        block(128),
        block(128),
        block(128),
        block(128),
        block(128),

        nn.BatchNorm2d(128),
        nn.AdaptiveAvgPool2d(1),
        Flatten(),
        nn.Linear(128, o_c),
    )


def eval(network, x, y_target):
    y = network(x)
    loss = nnf.cross_entropy(y, y_target)
    acc = (torch.argmax(y, dim=1) == y_target).sum() / len(y_target)
    return loss, acc, y


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


def train(network, opt, schedule, batch_size, max_batch_count, train_data, test_data, plotter):
    logger = Logger()

    for bi in itertools.count():
        if max_batch_count is not None and bi >= max_batch_count:
            break

        plotter.block_while_paused()
        logger.start_batch()

        if schedule is not None:
            lr = schedule(bi)
            logger.log("schedule", "lr", lr)
            for group in opt.param_groups:
                group["lr"] = lr

        network.eval()
        test_x, test_y_target = sample(test_data, batch_size)
        test_loss, test_acc, test_y = eval(network, test_x, test_y_target)

        logger.log("act", "output min", test_y.min())
        logger.log("act", "output max", test_y.max())
        logger.log("act", "output mean", test_y.mean())
        logger.log("act", "output std", test_y.std())

        network.train()
        train_x, train_y_target = sample(train_data, batch_size)
        train_loss, train_acc, _ = eval(network, train_x, train_y_target)

        opt.zero_grad(set_to_none=True)
        train_loss.backward()
        opt.step()

        logger.log("acc", "test", test_acc.item())
        logger.log("acc", "train", train_acc.item())
        logger.log("loss", "test", test_loss.item())
        logger.log("loss", "train", train_loss.item())

        for name, value in network.named_parameters(recurse=True):
            if "factor" in name:
                mag = value.abs()

                logger.log("scale", f"{name} min", mag.min())
                logger.log("scale", f"{name} mean", mag.mean())
                logger.log("scale", f"{name} max", mag.max())

        plotter.update(logger)

    return logger


def save_examples(x, bits: int):
    max_int = 2 ** bits - 1
    q = (x * max_int).clamp(0, max_int).int().float() / max_int
    save_image(q, "../ignored/binary_bit_inputs.png")


def main(plotter):
    train_data = dataset_to_tensors(CIFAR10("../ignored/data", train=True, download=True))
    test_data = dataset_to_tensors(CIFAR10("../ignored/data", train=False, download=True))

    bits = 8
    save_examples(train_data[0][:100], bits)

    (_, c_i, _, s_i) = train_data[0].shape
    o_c = train_data[1].max() + 1
    print(f"Image shape {c_i}x{s_i}x{s_i}, {o_c} categories")

    batch_size = 64
    max_batch_count = None

    network = build_network(s_i, c_i, o_c, bits, False, False, 0.0)
    network.to(DEVICE)

    # opt = SGD(network.parameters(), lr=0.0012, momentum=0.9, weight_decay=1e-3)
    opt = AdamW(network.parameters(), lr=2e-5)

    schedule = None
    # schedule = WarmupSchedule(100, FixedSchedule([0.1, 0.01, 0.001], [500, 500]))
    # schedule = FixedSchedule([0.002, 0.001, 0.0001], [1000, 200])
    # schedule = ExpSchedule(1e-6, 1e-3, 1000)

    train(network, opt, schedule, batch_size, max_batch_count, train_data, test_data, plotter)


if __name__ == '__main__':
    run_with_plotter(main)
    # main(DummyLogPlotter())
