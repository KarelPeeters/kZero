import itertools
import os

import torch.jit
from torch import nn
from torch.optim import AdamW

from lib.data.buffer import FileListSampler
from lib.data.file import DataFile
from lib.games import Game
from lib.logger import Logger
from lib.model.post_act import ResTower, ConcatInputsChannelwise, PredictionHeads, ScalarHead, ResBlock, ConvPolicyHead
from lib.networks import MuZeroNetworks
from lib.plotter import run_with_plotter, LogPlotter
from lib.schedule import LinearSchedule
from lib.train import TrainSettings, ScalarTarget
from lib.util import DEVICE


def main(plotter: LogPlotter):
    print(f"Using device {DEVICE}")

    game = Game.find("ttt")

    paths = [
        fr"C:\Documents\Programming\STTT\AlphaZero\data\all\ttt.json"
    ]
    files = [DataFile.open(game, p) for p in paths]

    include_final = True
    sampler = FileListSampler(game, files, batch_size=128, unroll_steps=5, include_final=include_final, threads=2)

    train = TrainSettings(
        game=game,
        value_weight=0.1,
        wdl_weight=1.0,
        policy_weight=1.0,
        moves_left_delta=20,
        moves_left_weight=0.0001,
        clip_norm=4.0,
        scalar_target=ScalarTarget.Final,
        train_in_eval_mode=False,
        mask_policy=False,
    )

    output_path = "../../data/muzero/all/unroll-again-nonorm"
    os.makedirs(output_path, exist_ok=False)

    channels = 64
    depth = 8
    saved_channels = 64

    representation = nn.Sequential(
        ResTower(depth, game.full_input_channels, channels, final_affine=False),
        # nn.LayerNorm([channels, game.board_size, game.board_size], elementwise_affine=False),
        nn.Hardtanh(-1.0, 1.0),
    )
    dynamics = ConcatInputsChannelwise(nn.Sequential(
        ResTower(depth, saved_channels + game.input_mv_channels, channels, final_affine=False),
        # nn.LayerNorm([channels, game.board_size, game.board_size], elementwise_affine=False),
        nn.Hardtanh(-1.0, 1.0),
        # Flip(dim=2),
    ))
    prediction = PredictionHeads(
        common=ResBlock(channels),
        scalar_head=ScalarHead(game.board_size, channels, 8, 128),
        # policy_head=AttentionPolicyHead(game, channels, 64)
        policy_head=ConvPolicyHead(game, channels),
    )

    networks = MuZeroNetworks(
        state_channels=channels,
        state_channels_saved=saved_channels,
        state_quant_bits=8,
        representation=representation,
        dynamics=dynamics,
        prediction=prediction,
    )
    networks.to(DEVICE)

    logger = Logger()

    schedule = LinearSchedule(1e-6, 1e-2, 200)
    optimizer = AdamW(networks.parameters(), weight_decay=1e-5)

    plotter.set_can_pause(True)

    print("Start training")
    for bi in itertools.count():
        if bi % 100 == 0:
            logger.save(f"{output_path}/log.npz")
        if bi % 500 == 0:
            torch.jit.save(torch.jit.script(networks), f"{output_path}/models_{bi}.pb")

        plotter.block_while_paused()
        print(f"bi: {bi}")
        logger.start_batch()

        # lr = schedule(bi)
        # logger.log("opt", "lr", lr)
        # for group in optimizer.param_groups:
        #     group["lr"] = lr

        batch = sampler.next_unrolled_batch()
        train.train_step(batch, networks, optimizer, logger)

        plotter.update(logger)


if __name__ == '__main__':
    run_with_plotter(main)
