import torch
import torch.nn.functional as nnf
from torch import nn

from lib.model.attention import multi_head_attention, init_xavier_normal_


class AttentionBlock(nn.Module):
    def __init__(
            self,
            self_attention: bool,
            d_input_q: int,
            d_input_kv: int,
            d_output: int,
            heads: int,
            d_k: int,
            d_v: int,
            beta: float = 1.0
    ):
        super().__init__()

        self.self_attention = self_attention
        if self_attention:
            assert d_input_q == d_input_kv

        self.d_input_q = d_input_q
        self.d_input_kv = d_input_kv
        self.d_output = d_output
        self.heads = heads
        self.d_k = d_k
        self.d_v = d_v
        d_kqv = 2 * d_k + d_v

        self.project_qkv = nn.Parameter(torch.empty(heads, d_kqv, d_input_q))

        if self_attention:
            self.project_q = self.project_qkv[:, :d_k, :]
            self.project_kv = self.project_qkv[:, d_k:, :]
        else:
            self.project_q = nn.Parameter(torch.empty(heads, d_k, d_input_q))
            self.project_kv = nn.Parameter(torch.empty(heads, d_k + d_v, d_input_kv))

        self.project_out = nn.Parameter(torch.empty(d_output, heads, d_v))

        # weight initialization according to DeepNorm
        project_k = self.project_kv[:, :d_k, :]
        project_v = self.project_kv[:, d_k:, :]

        init_xavier_normal_(self.project_q, d_input_q, d_k, gain=1)
        init_xavier_normal_(project_k, d_input_kv, d_k, gain=1)
        init_xavier_normal_(project_v, d_input_kv, d_v, gain=beta)
        init_xavier_normal_(self.project_out, heads * d_v, d_output, gain=beta)

    # TODO add test that asserts that results are identical if they're copies instead
    def forward_with_weights(self, input_q, input_kv):
        if self.self_attention:
            assert input_q is input_kv

        heads = self.heads
        d_k = self.d_k
        d_v = self.d_v
        d_input_q = self.d_input_q
        d_input_kv = self.d_input_kv
        d_kqv = 2 * d_k + d_v
        d_output = self.d_output

        (n, b, d_input_q1) = input_q.shape
        (m, b1, d_input_kv1) = input_kv.shape

        assert b == b1
        assert d_input_q1 == d_input_q
        assert d_input_kv1 == d_input_kv

        if self.self_attention:
            # fuse q anv kv projections in single operation
            qkv = nnf.linear(
                input_q.view(n * b, d_input_q),
                self.project_qkv.view(heads * d_kqv, d_input_q)
            ).view(n, b, heads, d_kqv)

            q = qkv[:, :, :, :d_k]
            k = qkv[:, :, :, d_k:2 * d_k]
            v = qkv[:, :, :, 2 * d_k:]
        else:
            # project q and kv separately
            q = nnf.linear(
                input_q.view(n * b, d_input_q),
                self.project_q.view(heads * d_k, d_input_q)
            ).view(n, b, heads, d_k)

            kv = nnf.linear(
                input_kv.view(n * b, d_input_kv),
                self.project_kv.view(heads * (d_k + d_v), d_input_kv)
            ).view(n, b, heads, d_k + d_v)

            k = kv[:, :, :, :d_k]
            v = kv[:, :, :, d_k:]

        att_result, weights = multi_head_attention(q, k, v)

        output_projected = nnf.linear(
            att_result.view(n * b, heads * d_v),
            self.project_out.view(d_output, heads * d_v)
        ).view(n, b, d_output)

        return output_projected, weights

    def forward(self, input_q, input_kv):
        result, _ = self.forward_with_weights(input_q, input_kv)
        return result
