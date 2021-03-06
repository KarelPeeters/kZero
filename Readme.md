# kZero

An implementation of the AlphaZero and MuZero papers for different board games, implemented in a combination of Python and Rust.

## Current status

The specific board games currently supported are Chess, Ataxx, SuperTicTacToe and TicTacToe. Almost all the code is generic over the specific game being played, so adding new 2-player games should be relatively very easy.

AlphaZero already works very well, especially for chess. For a rough idea after training for a couple of days on a RTX 3080, when playing Blitz on lichess it's approximately as strong as their "StockFish level 7".  MuZero is not really working yet more the more complicated games, it's still a work-in-progress.

## Overview

The neural network training is implemented in Python, and Rust is used to implement the more performance-critical selfplay.

During a training loop the training framework connects to the selfplay TCP server running on localhost and gives it some initial settings. The selfplay server writes generated games to a file, and when it has generated enough games it signals this to the training framework. The network is then trained on these new games, and when training finishes the new network is sent to the selfplay server.

Much effort has been put into optimizing selfplay, so network evaluations are batched and multiple threads and GPUs can be used, although the latter is not frequently tested. As is typical for my projects there is a lot of scope creep, so I ended up writing my own ONNX inference framework that directly calls into CuDNN combined with a couple of custom Cuda kernels to fill in the gaps.

## Project structure

The basic file structure is as follows:

### Python

The `python` folder contains the training code, we use the PyTorch framework. 

* `lib` contains the core training code, including neural network definitions and replay buffer
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