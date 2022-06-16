import torch
from torch import nn

from lib.layers.attention import AttentionBlock


class AttentionTower(nn.Module):
    def __init__(
            self,
            board_size: int, input_channels: int,
            depth: int,
            d_model: int, heads: int, d_k: int, d_v: int, d_ff: int,
            dropout: float
    ):
        super().__init__()
        self.board_size = board_size
        self.d_model = d_model

        alpha = (2 * depth) ** (1 / 4)
        beta = (8 * depth) ** (-1 / 4)

        self.expand = nn.Linear(input_channels, d_model, bias=False)
        self.embedding = nn.Parameter(torch.randn(board_size * board_size, d_model))

        self.encoders = nn.ModuleList(
            EncoderLayer(d_model, heads, d_k, d_v, d_ff, dropout, alpha=alpha, beta=beta)
            for _ in range(depth)
        )

    def forward(self, x):
        b, d_in, h, w = x.shape

        # "b c h w -> (h w) b c"
        shaped = x.permute(2, 3, 0, 1).view(h * w, b, d_in)

        expanded = self.expand(shaped.reshape(h * w * b, d_in)).view(h * w, b, self.d_model)
        curr = expanded + self.embedding.unsqueeze(1)

        for encoder in self.encoders:
            curr, _ = encoder(curr)

        # "(h w) b c -> b c h w"
        reshaped = curr.view((h, w, b, self.d_model)).permute((2, 3, 0, 1))
        return reshaped


class EncoderLayer(nn.Module):
    def __init__(
            self,
            d_model: int, heads: int,
            d_k: int, d_v: int, d_ff: int,
            dropout: float,
            alpha: float = 1.0, beta: float = 1.0
    ):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.alpha = alpha

        self.att = AttentionBlock(
            self_attention=True,
            d_input_q=d_model, d_input_kv=d_model, d_output=d_model,
            heads=heads, d_k=d_k, d_v=d_v,
            beta=beta
        )

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False),
        )

        self.dropout = nn.Dropout(dropout)
        self.norm_att = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm_ff = nn.LayerNorm(d_model, elementwise_affine=False)

    def forward(self, input):
        (n, b, d_model) = input.shape
        assert d_model == self.d_model

        att_inner, weights = self.att(input, input)
        att_result = self.norm_att(self.dropout(att_inner) + self.alpha * input)

        ff_inner = self.ff(att_result.view(n * b, d_model)).view(n, b, d_model)
        ff_result = self.norm_ff(self.dropout(ff_inner) + self.alpha * input)

        return ff_result, weights
