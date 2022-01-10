import torch
import torch.nn.functional as nnf
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable


class BitSignFunction(Function):
    # noinspection PyMethodOverriding
    @staticmethod
    def forward(ctx, x, clamp_grad: bool):
        assert x.dtype == torch.float
        ctx.save_for_backward(x)
        ctx.clamp_grad = clamp_grad
        return torch.where(x > 0, 1.0, -1.0)

    @staticmethod
    @once_differentiable
    def backward(ctx, y_grad):
        x, = ctx.saved_tensors
        clamp_grad = ctx.clamp_grad

        if clamp_grad:
            return (x.abs() < 1) * y_grad, None
        else:
            return y_grad, None


class BSign(nn.Module):
    def __init__(self, clamp_grad: bool):
        super().__init__()
        self.clamp_grad = clamp_grad

    def forward(self, x):
        return BitSignFunction.apply(x, self.clamp_grad)


class BLinear(nn.Module):
    def __init__(self, size_in: int, size_out: int, clamp_grad: bool):
        super().__init__()
        self.clamp_grad = clamp_grad

        self.div = size_in ** .5
        self.weight = nn.Parameter(torch.normal(0.0, 0.01, (size_out, size_in)))

    def forward(self, x):
        weight_bin = BitSignFunction.apply(self.weight, self.clamp_grad)
        return nnf.linear(x, weight_bin) / self.div


class BConv2d(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, filter_size: int, clamp_grad: bool):
        super().__init__()
        self.clamp_grad = clamp_grad

        assert filter_size % 2 == 1, f"Filter size must be odd, got {filter_size}"
        self.padding = filter_size // 2

        self.div = (channels_in * filter_size * filter_size) ** .5
        self.weight = nn.Parameter(torch.normal(0.0, 0.01, (channels_out, channels_in, filter_size, filter_size)))

    def forward(self, x):
        weight_bin = BitSignFunction.apply(self.weight, self.clamp_grad)
        return nnf.conv2d(x, weight_bin, padding=self.padding) / self.div
