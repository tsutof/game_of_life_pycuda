# game_of_life_pycuda
**An Implementation of Conway's Game of Life with PyCUDA**

<img src="./screencast.gif" alt="Screencast" title="Screencast">

## What does this application do?

This application executes Conway's Game of Life accelerated by PyCUDA.

## Prerequisites

CUDA-enabled system
This application is tested with NVIDIA Jetson Nano Developer Kit.

## Installation

- Add the CUDA toolkit path. (Add the following setting to .bashrc)

```
export PATH=$PATH:/usr/local/cuda/bin
```

- Install the pip3 tool.

```
$ sudo apt-get update
$ sudo apt-get install python3-pip
```

- Install the NumPy package and the PyCUDA package.

```
$ pip3 install numpy --user
$ pip3 install pycuda --user
```

- Install this application.

```
```
