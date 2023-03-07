from typing import Optional

import einops
import torch
from fairscale.nn import checkpoint_wrapper
from perceiver.model.core import PerceiverEncoder, InputAdapter, OutputAdapter, \
    TrainableQueryProvider, QueryProvider, CrossAttentionLayer, init_parameters
from torch import nn

from lib.data.file import DataFile
from lib.data.position import PositionBatch
from lib.games import Game


class ChessInputAdapter(InputAdapter):
    def __init__(self, d: int):
        super().__init__(d)
        self.d = d

        # square
        self.token_square = nn.Parameter(torch.randn(8, 8, d))
        # none, ..us_piece, ..other_piece
        self.token_piece = nn.Parameter(torch.randn(1 + 2 * 6, d))
        # false, true
        self.token_en_passant = nn.Parameter(torch.randn(2, d))
        # all possible combinations of castling rights
        self.token_castle = nn.Parameter(torch.randn(2 ** 4, d))

    def forward(self, batch):
        device = batch.device
        b, c, h, w = batch.shape

        scalars = batch[:, :8, 0, 0]
        bools = batch[:, 8:, :, :]
        castle_rights = scalars[:, 2:6]
        en_passant = bools[:, -1, :, :]

        piece_index = torch.argmax(torch.cat([
            torch.full((1, 1, 1, 1), 0.5, device=device).expand((b, 1, h, w)),
            bools[:, :2 * 6, :, :],
        ], dim=1), dim=1)
        input_pieces = self.token_square.unsqueeze(0) + self.token_piece[piece_index, :]

        input_en_passant = self.token_en_passant[en_passant.long()]

        castle_values = torch.tensor([8, 4, 2, 1], dtype=torch.long, device=device)
        castle_index = (castle_rights.int() * castle_values).sum(-1)
        input_castle = self.token_castle[castle_index, :]

        # (b, n, d)
        input = torch.cat([
            einops.rearrange(input_pieces + input_en_passant, "b h w d -> b (h w) d"),
            einops.rearrange(input_castle, "b d -> b 1 d"),
        ], dim=1)
        return input


class UnitOutputAdapter(OutputAdapter):
    def forward(self, x):
        return x


class DoubleDecoder(nn.Module):
    def __init__(
            self,
            output_adapter: OutputAdapter,
            output_query_provider: QueryProvider,
            num_latent_channels: int,
            num_cross_attention_heads: int = 4,
            num_cross_attention_qk_channels: Optional[int] = None,
            num_cross_attention_v_channels: Optional[int] = None,
            cross_attention_widening_factor: int = 1,
            cross_attention_residual: bool = True,
            dropout: float = 0.0,
            init_scale: float = 0.02,
            activation_checkpointing: bool = False,
            activation_offloading: bool = False,
    ):
        super().__init__()

        self.output_query_provider = output_query_provider
        self.output_adapter = output_adapter

        cross_attn0 = CrossAttentionLayer(
            num_heads=num_cross_attention_heads,
            num_q_input_channels=output_query_provider.num_query_channels,
            num_kv_input_channels=num_latent_channels,
            num_qk_channels=num_cross_attention_qk_channels,
            num_v_channels=num_cross_attention_v_channels,
            widening_factor=cross_attention_widening_factor,
            attention_residual=cross_attention_residual,
            dropout=dropout,
        )
        cross_attn1 = CrossAttentionLayer(
            num_heads=num_cross_attention_heads,
            num_q_input_channels=output_query_provider.num_query_channels,
            num_kv_input_channels=num_latent_channels,
            num_qk_channels=num_cross_attention_qk_channels,
            num_v_channels=num_cross_attention_v_channels,
            widening_factor=cross_attention_widening_factor,
            attention_residual=cross_attention_residual,
            dropout=dropout,
        )

        if activation_checkpointing:
            cross_attn0 = checkpoint_wrapper(cross_attn0, offload_to_cpu=activation_offloading)
            cross_attn1 = checkpoint_wrapper(cross_attn1, offload_to_cpu=activation_offloading)

        self.cross_attn0 = cross_attn0
        self.cross_attn1 = cross_attn1
        self._init_parameters(init_scale)

    def _init_parameters(self, init_scale: float):
        with torch.no_grad():
            init_parameters(self, init_scale)

    def forward(self, x_latent, x_adapted=None, **kwargs):
        output_query = self.output_query_provider(x_adapted)
        output0 = self.cross_attn0(output_query, x_latent)
        output1 = self.cross_attn1(output0, x_latent)
        return self.output_adapter(output1, **kwargs)


# TODO combine castling tokens
# TODO add counters as inputs
# TODO init scale?
class ChessPerceiverModel(nn.Module):
    def __init__(self, game: Game, d: int, heads: int, latents: int, count_cross: int, count_self: int,
                 checkpointing: bool):
        super().__init__()
        assert game.name == "chess"

        d_input = d
        d_latent = d
        d_output = d

        num_chess_moves = 1880

        self.encoder = PerceiverEncoder(
            input_adapter=ChessInputAdapter(d=d_input),
            num_latents=latents,
            num_latent_channels=d_latent,
            num_cross_attention_heads=heads,
            num_cross_attention_qk_channels=None,
            num_cross_attention_v_channels=None,
            num_cross_attention_layers=count_cross,
            first_cross_attention_layer_shared=False,
            cross_attention_widening_factor=1,
            num_self_attention_heads=heads,
            num_self_attention_qk_channels=None,
            num_self_attention_v_channels=None,
            # TODO difference between blocks and layers?
            num_self_attention_layers_per_block=1,
            num_self_attention_blocks=count_cross + count_self,
            first_self_attention_block_shared=False,
            self_attention_widening_factor=1,
            dropout=0.0,
            init_scale=0.02,
            activation_checkpointing=checkpointing,
            activation_offloading=False,
        )

        self.decoder = DoubleDecoder(
            output_adapter=UnitOutputAdapter(),
            output_query_provider=TrainableQueryProvider(num_chess_moves + 1, d_output),
            num_latent_channels=d_latent,
            num_cross_attention_heads=heads,
            num_cross_attention_qk_channels=None,
            num_cross_attention_v_channels=None,
            cross_attention_widening_factor=1,
            # TODO this doesn't really make sense, we're just providing a policy residual?
            cross_attention_residual=True,
            dropout=0.0,
            init_scale=0.02,
            activation_checkpointing=checkpointing,
            activation_offloading=False,
        )

        self.flat_policy = nn.Linear(d_output, 1)
        self.flat_scalars = nn.Linear(d_output, 5)

    def forward(self, batch):
        encoded = self.encoder(batch)
        decoded_full = self.decoder(encoded)

        policy = self.flat_policy(decoded_full[:, :-1, :]).squeeze(-1)
        scalars = self.flat_scalars(decoded_full[:, -1, :])

        return scalars, policy


def main():
    game = Game.find("chess")
    model = ChessPerceiverModel(game, 128, 8, 64 + 1, 2, 6, False)

    device = "cpu"
    model.to(device)

    path = r"C:\Documents\Programming\STTT\kZero\data\loop\chess\16x128_pst\selfplay\games_2400.json"
    file = DataFile.open(game, path)
    # data = DataGroup.from_files(game, [file], 0, 1)
    # sampler = PositionSampler(data, 16, None, False, False, False, 1)
    # batch = sampler.next_batch()
    # sampler.close()

    batch = PositionBatch(game, list(file.positions[:16]), False, False)
    scalars, policy = model(batch.input_full.to(device))
    print(scalars.shape)
    print(policy.shape)

    # (B, N, D)
    # x = torch.randn(16, 64, 128)
    # y = model(x)
    # print(y.shape)


if __name__ == '__main__':
    main()
