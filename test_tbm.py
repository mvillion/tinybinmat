#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
echo_asm convolution test.

Usage:
    test_tbm [--dummy-exit] [--n-run=<n>] [--n-run-step=<n>]

Options:
    -h --help             show this screen
    --dummy-exit          exit for debugger
    --n-run=<n>           total number of runs [default: 10]
    --n-run-step=<n>      sub-step run (for L3 and L2 cache)
"""
import tinybinmat
import numpy as np
import sys

if __name__ == "__main__":
    from docopt import docopt
    arg = docopt(__doc__)
    if arg["--dummy-exit"]:
        print("dummy exit")
        sys.exit()

    np.random.seed(0)
    n_run = int(arg["--n-run"])
    try:
        n_run_step = int(arg["--n-run-step"])
    except TypeError:
        n_run_step = n_run

    n_bit = 6
    mat = np.random.randint(0, 2**n_bit-1, (n_run, 8), dtype=np.uint8)
    mat[:, n_bit:] = 0

    n_print = 2
    mat_print = mat[:n_print, :]
    tinybinmat.print(mat_print, n_bit, " x")

    mat8 = tinybinmat.sprint(mat, n_bit, np.arange(2, dtype=np.uint8))
    print(mat8[:n_print, :, :])

    mats = tinybinmat.sprint(mat_print, n_bit, np.frombuffer(b" x", np.uint8))
    mats = mats.view("S%d" % n_bit).reshape(mats.shape[:-1])
    print(mats)

    mat8t = tinybinmat.transpose(mat, n_bit)
    mat8t = tinybinmat.sprint(mat8t, n_bit, np.arange(2, dtype=np.uint8))
    ok = np.array_equal(mat8.transpose(0, 2, 1), mat8t)
    print("transpose8 is %sok" % ["not ", ""][ok])

    n_bit = 10
    mat = np.random.randint(0, 2**n_bit-1, (n_run, 16), dtype=np.uint16)
    mat[:, n_bit:] = 0
    mat_print = mat[:n_print, :]
    tinybinmat.print(mat_print, n_bit, " x")

    mat16 = tinybinmat.sprint(mat, n_bit, np.arange(2, dtype=np.uint8))
    print(mat16[:n_print, :, :])

    mat16t = tinybinmat.transpose(mat, n_bit)
    mat16t = tinybinmat.sprint(mat16t, n_bit, np.arange(2, dtype=np.uint8))
    ok = np.array_equal(mat16.transpose(0, 2, 1), mat16t)
    print("transpose16 is %sok" % ["not ", ""][ok])
