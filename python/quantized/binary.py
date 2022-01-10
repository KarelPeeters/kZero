import torch
import torch.nn.functional as nnf
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable


class BitSignFunction(Function):
    # noinspection PyMethodOverriding
    @staticmethod
    def forward(ctx, x):
        assert x.dtype == torch.float
        ctx.save_for_backward(x)
        return torch.where(x > 0, 1.0, -1.0)

    @staticmethod
    @once_differentiable
    def backward(ctx, y_grad):
        x, = ctx.saved_tensors
        return (x.abs() < 1) * y_grad


class BSign(nn.Module):
    @staticmethod
    def forward(x):
        return BitSignFunction.apply(x)


class BLinear(nn.Module):
    def __init__(self, size_in: int, size_out: int):
        super().__init__()

        self.weight = nn.Parameter(torch.normal(0.0, 0.1, (size_out, size_in)))

    def forward(self, x):
        return nnf.linear(x, BitSignFunction.apply(self.weight))
