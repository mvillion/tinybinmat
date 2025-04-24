#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
echo_asm convolution test.

Usage:
    test_tbm [--dummy-exit] [--n-run=<n>] [--n-run-step=<n>]

Options:
    -h --help             show this screen
    --dummy-exit          exit for debugger
    --n-run=<n>           total number of runs [default: 2]
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
    tinybinmat.print(mat, n_bit, " x")

    mat8 = tinybinmat.sprint(mat, n_bit, np.arange(2, dtype=np.uint8))
    print(mat8)

    mats = tinybinmat.sprint(mat, n_bit, np.frombuffer(b" x", np.uint8))
    mats = mats.view("S%d" % n_bit).reshape(mats.shape[:-1])
    print(mats)

    n_bit = 10
    mat = np.random.randint(0, 2**n_bit-1, (n_run, 16), dtype=np.uint16)
    mat[:, n_bit:] = 0
    tinybinmat.print(mat, n_bit, " x")

    mat16 = tinybinmat.sprint(mat, n_bit, np.arange(2, dtype=np.uint8))
    print(mat16)
