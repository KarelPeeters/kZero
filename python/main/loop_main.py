import shutil

from torch.optim import AdamW

from lib.games import Game
from lib.loop import FixedSelfplaySettings, LoopSettings
from lib.model.lc0_pre_act import LCZOldPreNetwork
from lib.model.simple import DenseNetwork
from lib.selfplay_client import SelfplaySettings
from lib.train import TrainSettings


def main():
    game = Game.find("chess")
    print(f"Using game {game.name}")

    fixed_settings = FixedSelfplaySettings(
        game=game,
        threads_per_device=2,
        batch_size=256,
        games_per_gen=100,
    )

    dirichlet_alpha = 10 / game.average_available_moves
    print(f"Using dirichlet alpha {dirichlet_alpha}")

    selfplay_settings = SelfplaySettings(
        temperature=1.0,
        # TODO alphazero uses value 30 (plies, 15 moves)
        zero_temp_move_count=20,
        # TODO give this information to zero tree search too! now it might be stalling without a good reason
        max_game_length=300,
        keep_tree=False,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_eps=0.25,
        full_search_prob=1.0,
        # TODO increase this again?
        full_iterations=600,
        part_iterations=600,
        exploration_weight=2.0,
        random_symmetries=True,
        cache_size=0,
    )

    train_settings = TrainSettings(
        game=game,
        value_weight=0.1,
        wdl_weight=0.1,
        policy_weight=1.0,
        batch_size=256,
        clip_norm=20.0,
    )

    def initial_network():
        return LCZOldPreNetwork(game, 4, 32, 32, True, 8, 128)

    settings = LoopSettings(
        root_path=f"data/new_loop_real/first/{game.name}/",
        initial_network=initial_network,

        target_buffer_size=100_000,
        train_steps_per_gen=2,

        optimizer=lambda params: AdamW(params, weight_decay=1e-5),

        fixed_settings=fixed_settings,
        selfplay_settings=selfplay_settings,
        train_settings=train_settings,
    )

    shutil.rmtree(settings.root_path)

    print_expected_buffer_behaviour(settings, game.average_game_length)

    settings.run_loop()


def print_expected_buffer_behaviour(settings: LoopSettings, average_game_length: int):
    games_in_buffer = settings.target_buffer_size / average_game_length
    gens_in_buffer = games_in_buffer / settings.fixed_settings.games_per_gen

    positions_per_gen = settings.train_steps_per_gen * settings.train_settings.batch_size
    visits_per_position = gens_in_buffer * positions_per_gen / settings.target_buffer_size
    visits_per_game = visits_per_position * average_game_length

    print("Expected numbers:")
    print(f"  Positions in buffer: {settings.target_buffer_size}")
    print(f"  Games in buffer: {games_in_buffer}")
    print(f"  Generations in buffer: {gens_in_buffer}")
    print(f"  Positions per gen: {positions_per_gen}")
    print(f"  Visits per position: {visits_per_position}")
    print(f"  Visits per game: {visits_per_game}")


if __name__ == '__main__':
    main()
