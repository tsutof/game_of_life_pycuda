#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# MIT License
#
# Copyright (c) 2019 Tsutomu Furuse
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

'''
An Implementation of Conway's Game of Life with PyCUDA

'''

import numpy as np
import curses
from curses import wrapper
import time
import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule
import argparse

get_cell_value = lambda buf, height, width, row, col : \
    buf[row % height, col % width]

row_string = lambda row_array : \
    ''.join(['O' if c != 0 else ' ' for c in row_array])

def print_state(stdscr, gen, state, is_nodelay, info=''):
    stdscr.clear()
    stdscr.nodelay(is_nodelay)
    live_count = 0
    height, width = state.shape
    for row in range(height):
        stdscr.addstr(row, 0, row_string(state[row]))
        live_count += np.sum(state[row])
    ret = True
    msg = None
    if is_nodelay:
        msg = 'Q to Quit'
    else:
        msg = 'Hit Any Key to Continue'
    stdscr.addstr(height, 0, \
        '(%d x %d)  Gen:%6d  Lives:%6d  %s  [%s] ' \
        % (height, width, gen, live_count, info, msg), \
        curses.A_REVERSE)
    key = stdscr.getch()
    if is_nodelay:
        if key == ord('q') or key == ord('Q'):
            ret = False
    stdscr.refresh()
    return ret

def get_next_state_gpu(state, next_state):
    height, width = state.shape

    mod = SourceModule("""
        __global__ void get_next_state(int *state, int *nextState, int height, int width)
        {
            unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
            unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
            unsigned int idx = iy * width + ix;
            int sum = 0;
            int val, nextVal;

            if (ix >= width || iy >= height) {
                return;
            }

            val = state[idx];

            sum += state[((iy - 1) % height) * width + ((ix - 1) % width)];
            sum += state[((iy - 1) % height) * width + (ix       % width)];
            sum += state[((iy - 1) % height) * width + ((ix + 1) % width)];
            sum += state[(iy       % height) * width + ((ix - 1) % width)];
            sum += state[(iy       % height) * width + ((ix + 1) % width)];
            sum += state[((iy + 1) % height) * width + ((ix - 1) % width)];
            sum += state[((iy + 1) % height) * width + (ix       % width)];
            sum += state[((iy + 1) % height) * width + ((ix + 1) % width)];

            if (val == 0 && sum == 3) {
                nextVal = 1;
            }
            else if (val != 0 && (sum >= 2 && sum <= 3)) {
                nextVal = 1;
            }
            else {
                nextVal = 0;
            }
            nextState[idx] = nextVal;
        }
        """)
    kernel_func = mod.get_function("get_next_state")

    blk_dim = (32, 32, 1)
    grid_dim = ((width + blk_dim[0] - 1) // blk_dim[0], \
        (height + blk_dim[1] - 1) // blk_dim[1], 1)
    kernel_func(
        drv.In(state), drv.Out(next_state), \
        np.int32(height), np.int32(width), \
        block=blk_dim, grid=grid_dim)

def get_next_cell_state(state, next_state, row, col):
    height, width = state.shape
    c = get_cell_value(state, height, width, row, col)
    sum = 0
    sum += get_cell_value(state, height, width, row - 1, col - 1)
    sum += get_cell_value(state, height, width, row - 1, col)
    sum += get_cell_value(state, height, width, row - 1, col + 1)
    sum += get_cell_value(state, height, width, row,     col - 1)
    sum += get_cell_value(state, height, width, row,     col + 1)
    sum += get_cell_value(state, height, width, row + 1, col - 1)
    sum += get_cell_value(state, height, width, row + 1, col)
    sum += get_cell_value(state, height, width, row + 1, col + 1)
    if c == 0 and sum == 3:
        val = 1
    elif c == 1 and sum in (2, 3):
        val = 1
    else:
        val = 0
    next_state[row, col] = val

def get_next_state_cpu(state, next_state):
    height, width = state.shape
    for row in range(height):
            for col in range(width):
                get_next_cell_state(state, next_state, row, col)

def run_loop(stdscr, *args):
    cpu_flag = args[0]
    term_height, term_width = stdscr.getmaxyx()
    height = term_height - 1
    width = term_width
    next_state = np.empty((height, width), dtype=np.int32)
    state = np.random.randint(2, size=(height, width), dtype=np.int32)
    gen = 0
    is_nodelay = False
    if cpu_flag:
        info = '<CPU Mode>'
    else:
        info = '<GPU Mode>'

    while True:
        ret = print_state(stdscr, gen, state, is_nodelay, info)
        if not ret:
            break

        if cpu_flag:
            start_time = time.time()
            get_next_state_cpu(state, next_state)
            elapsed_time = time.time() - start_time
        else:
            start_time = time.time()
            get_next_state_gpu(state, next_state)
            elapsed_time = time.time() - start_time
        info = 'Time: %.6f' % (elapsed_time)

        state, next_state = next_state, state
        is_nodelay = True
        gen += 1

def main():
    parser = argparse.ArgumentParser(description='Life Game Accelerated by CUDA')
    parser.add_argument('--cpu', action='store_true', \
        help='Run by CPU')
    args = parser.parse_args()
    wrapper(run_loop, args.cpu)
    print('Done')

if __name__ == "__main__":
    main()