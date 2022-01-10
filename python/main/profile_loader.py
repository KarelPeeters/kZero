import time

from lib.data.buffer import FileListSampler
from lib.data.file import DataFile
from lib.games import Game


def main():
    game = Game.find("chess")
    file = DataFile.open(game, "../../data/lichess/09")
    loader = FileListSampler(game, [file], 4096, 4)

    total = 0
    start = time.perf_counter()

    while True:
        total += len(loader.next_batch())
        delta = time.perf_counter() - start
        throughput = total / delta
        print(f"Throughput: {throughput}")


if __name__ == '__main__':
    main()
