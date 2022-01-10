import itertools
from typing import Optional

import torch
import torchvision
from torch import nn
from torch.nn import functional as nnf
from torch.optim import SGD
from torchvision.transforms import ToTensor

from lib.logger import Logger
from lib.plotter import run_with_plotter, LogPlotter
from quantized.quant_bits import QuantModule, QLinear

DEVICE = "cpu"


def build_network(bits, scale):
    return nn.Sequential(
        nn.Flatten(),
        QLinear(nn.Linear(28 * 28, 256), QuantModule(bits, scale), QuantModule(bits, scale)),
        nn.ReLU(),
        QLinear(nn.Linear(256, 10), QuantModule(bits, scale), QuantModule(bits, scale)),
    )


def eval(network, x, y_target):
    y = network(x)
    loss = nnf.cross_entropy(y, y_target)
    acc = (torch.argmax(y, dim=1) == y_target).sum() / len(y_target)
    return loss, acc


def sample(dataset, count: int):
    i = torch.randint(len(dataset), (count,))
    return dataset.data[i].float().to(DEVICE) / 255, dataset.targets[i].to(DEVICE)


def main(plotter: Optional[LogPlotter]):
    train_data = torchvision.datasets.MNIST("../ignored/data", train=True, download=True, transform=ToTensor())
    test_data = torchvision.datasets.MNIST("../ignored/data", train=False, download=True, transform=ToTensor())

    bits = torch.tensor(16)
    network = build_network(bits, 0.1)
    network.to(DEVICE)

    opt = SGD(network.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-2)
    batch_size = 256

    logger = Logger()

    for bi in itertools.count():
        plotter.block_while_paused()

        if bi % 800 == 0:
            bits.fill_(max(1, bits.item() / 2))
            print(f"bits: {bits.item()}")
            logger.log("settings", "bits", bits.item())

        network.eval()
        test_x, test_y_target = sample(test_data, batch_size)
        test_loss, test_acc = eval(network, test_x, test_y_target)

        network.train()
        train_x, train_y_target = sample(train_data, batch_size)
        train_loss, train_acc = eval(network, train_x, train_y_target)

        opt.zero_grad(set_to_none=True)
        train_loss.backward()
        opt.step()

        logger.log("acc", "test", test_acc.item())
        logger.log("acc", "train", train_acc.item())
        logger.log("loss", "test", test_loss.item())
        logger.log("loss", "train", train_loss.item())

        for (mi, module) in enumerate(network.modules()):
            if isinstance(module, QLinear):
                logger.log("scale", f"{mi} w min", module.linear.weight.min())
                logger.log("scale", f"{mi} w mean", module.linear.weight.mean())
                logger.log("scale", f"{mi} w max", module.linear.weight.max())

                if module.linear.bias is not None:
                    logger.log("scale", f"{mi} b", module.linear.bias.min())
                    logger.log("scale", f"{mi} b", module.linear.bias.mean())
                    logger.log("scale", f"{mi} b", module.linear.bias.min())

        if plotter:
            plotter.update(logger)


if __name__ == '__main__':
    run_with_plotter(main)
