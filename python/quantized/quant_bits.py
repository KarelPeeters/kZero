import torch.nn.functional as nnf
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable


class QuantFunc(Function):
    # noinspection PyMethodOverriding
    @staticmethod
    def forward(ctx, x, bits: int, half_scale: float):
        if bits == 1:
            return (x >= 0).float()
        else:
            exp = bits - 1
            scale = 2 ** exp / half_scale

            return (x * scale).int().clamp(-2 ** exp, 2 ** exp - 1).float() / scale

    @staticmethod
    @once_differentiable
    def backward(ctx, y_grad):
        # TODO stop grad for values too far out of bounds
        return y_grad, None, None


class QuantModule(nn.Module):
    def __init__(self, bits: int, half_scale: float):
        super().__init__()
        self.bits = bits
        self.half_scale = half_scale

    def forward(self, x):
        return QuantFunc.apply(x, self.bits, self.half_scale)


class QLinear(nn.Module):
    def __init__(self, linear: nn.Linear, q_weight: QuantModule, q_bias: QuantModule):
        super().__init__()

        self.linear = linear
        self.q_weight = q_weight
        self.q_bias = q_bias

    def forward(self, x):
        return nnf.linear(
            x,
            self.q_weight(self.linear.weight),
            self.q_bias(self.linear.bias) if self.linear.bias is not None else None,
        )
