# kZero

A from-scratch implementation of the [AlphaGo](https://www.nature.com/articles/nature24270), [AlphaZero](https://arxiv.org/abs/1712.01815) and [MuZero](https://www.nature.com/articles/s41586-020-03051-4.epdf) papers for different board games, written in a combination of Python and Rust.

For GPU inference the [Kyanite library](https://github.com/KarelPeeters/Kyanite) is used.

## Table of contents

<!-- TOC -->
* [kZero](#kzero)
  * [Table of contents](#table-of-contents)
  * [Project Overview](#project-overview)
    * [Top level](#top-level)
    * [Selfplay server (Rust)](#selfplay-server-rust)
    * [Training (Python)](#training-python)
  * [Current status and results](#current-status-and-results)
  * [File structure](#file-structure)
    * [Python](#python)
    * [Rust](#rust)
      * [Core crates](#core-crates)
      * [Low-level crates](#low-level-crates)
      * [Utility crates](#utility-crates)
<!-- TOC -->

## Project Overview

### Top level

The top-level overview is shown in the figure above. The neural network training is implemented in Python, and Rust is used to implement the more performance-critical selfplay. The two parts communicate though a TCP connection and the file system.

![Top level system diagram](./docs/arch_overview.svg)

During a training loop the training framework connects to the selfplay TCP server running on localhost and gives it some initial settings. The selfplay server writes generated games to a file, and when it has generated enough games it signals this to the training framework. The network is then trained on these new games, and when training finishes the new network is sent to the selfplay server, closing the loop.

### Selfplay server (Rust)

The selfplay server is shown in the figure below. The orange boxes are the different threads running in the system.

![Selfplay server diagram](./docs/arch_selfplay.svg)

* The _commander_ receives commands from the TCP socket and forwards them to the appropriate place. Selfplay settings and hyperparameters are sent to the executors, and new neural networks are loaded from the filesystem and sent to the executors.
* The _generator_ thread pools run multiple simulations concurrently. Each simulation has its own MCTS tree that is grown over time. NN evaluation requests are sent to an executor.
* The _executor_ threads collect NN evaluations requests into batches and use the NN inference framework described later to run these evaluations, sending results back to the corresponding generator.
* Finally the _collector_ receives finished simulations from the generators, writes them to a file and notifies the TCP socket once enough simulations have finished.

Much effort was put into optimizing this system:

* Rust is used to get good baseline performance for the tricky code in tree search, memory transfers, thread interactions, NN input encoding, ...
* The MCTS itself uses virtual loss to collect small batches of evaluations.
* Many simulations are run concurrently on async thread pools. This allows for a second level of batching in the executors.
* Each GPU can have multiple executor threads, enabling concurrent batching, memory transfers and execution.
* Multiple GPUs can be used at full speed.

### Training (Python)

Finally the training component is shown in the figure below.

![Training diagram](./docs/arch_training.svg)

The python framework manages the user configuration, and sends the relevant parameters and the current neural network to the selfplay server. 

Incoming data files are added to the replay buffer. The sampler uses multiple threads to load training data from those data files, building batches. These batches are used in NN training. Finally newly trained networks are sent to the selfplay server, closing the loop.

The status of the replay buffer and training statistics are plotted in realtime using a Qt GUI.


## Current status and results

The AlphaZero implementation fully works. The specific board games that have had training runs are Chess, Ataxx and Go, all relatively successful. These training runs typically last for a couple of days on modest hardware ~2x GTX 3090.

Almost all the code is generic over the specific game being played, so adding new games or even more exotic RL environments should be easy.

MuZero is not working yet, there is some training instability that still has to be fixed.

## File structure

The basic file structure is as follows:

### Python

The `python` folder contains the training code, we use the PyTorch framework. 

* `lib` contains the training framework
* `lib/data` contains data file loading, the replay buffer, the sampler, ...
* `lib/mapping` contains metadata for building the NN architecture for various games
* `lib/lib` contains metadata for building the NN architecture for various games
* `lib/model` contains building blocks for NN architectures
* `main` contains the entry points for starting supervised training or selfplay

### Rust

The `rust` folder is a workspace consisting of a bunch of crates:

#### Core crates

* `kz-core` is the most important crate, it contains the AlphaZero and MuZero tree search implementations, game-specific details and NN wrappers.
* `kz-selfplay` is the selfplay server used during training loops. It runs a bunch of games in parallel on an async thread pool.
* `kz-misc` contains various utilities for evaluating network performance, converting between file formats, and is the crate where I do most of my experimenting in.
* `kz-lichess` is a lichess bot wrapper, you can occasionally play against it here: https://lichess.org/@/kZero-bot

#### Low-level crates

* `cuda-sys` contains Rust wrappers for Cuda and related frameworks, most importantly CuDNN. These wrappers are generated based on the system headers at build time.
* `nn-graph` is a graph format for NN inference, along with an ONNX parser and simple CPU executor.
* `cuda-nn-eval` is a Cuda/CuDNN executor for NN inference, based on the previously mentioned I

#### Utility crates

* `kz-util`: Boring utility functions that don't fit anywhere else.
* `pgn-reader`: A streaming pgn file parser. This was initially useful to pretrain networks on the lichess and CCRL databases.
* `licorice`: A "fork" of https://gitlab.com/hyperchessbotauthor/licorice with some small updates to get it working on lichess again.