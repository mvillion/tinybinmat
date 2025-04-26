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

from time import time


def test_transpose(mat, n_bit):
    mat8 = tinybinmat.sprint(mat, n_bit, np.arange(2, dtype=np.uint8))

    t0 = time()
    ref = np.ascontiguousarray(mat8.transpose(0, 2, 1))
    ref_duration = time()-t0

    t0 = time()
    mat8t = tinybinmat.transpose(mat)
    duration = time()-t0

    mat8t = tinybinmat.sprint(mat8t, n_bit, np.arange(2, dtype=np.uint8))
    ok = np.array_equal(ref, mat8t)
    if ok:
        print("transpose%d is ok" % (mat.itemsize*8))
    else:
        raise RuntimeError("error for transpose%d" % (mat.itemsize*8))
    print("duration vs ref: %f" % (duration/ref_duration))

    return mat8


def test_mult_t(mat, matb, n_bit):
    mat8 = tinybinmat.sprint(mat, n_bit, np.arange(2, dtype=np.uint8))
    matb8 = tinybinmat.sprint(matb, n_bit, np.arange(2, dtype=np.uint8))

    t0 = time()
    ref = mat8 @ matb8.transpose(0, 2, 1)
    ref &= 1
    ref_duration = time()-t0

    t0 = time()
    prod = tinybinmat.mult_t(mat, matb)
    duration = time()-t0

    prod = tinybinmat.sprint(prod, n_bit, np.arange(2, dtype=np.uint8))
    ok = np.array_equal(ref, prod.transpose(0, 2, 1))
    if ok:
        print("mult_t%d is ok" % (mat.itemsize*8))
    else:
        raise RuntimeError("error for mult_t%d" % (mat.itemsize*8))
    print("duration vs ref: %f" % (duration/ref_duration))

    return mat8


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
    mat2 = np.random.randint(0, 2**n_bit-1, (n_run, 8), dtype=np.uint8)
    mat2[:, n_bit:] = 0

    n_print = 2
    mat_print = mat[:n_print, :]
    tinybinmat.print(mat_print, n_bit, " x")

    mat8 = test_transpose(mat, n_bit)
    print(mat8[:n_print, :, :])

    test_mult_t(mat, mat2, n_bit)

    mats = tinybinmat.sprint(mat_print, n_bit, np.frombuffer(b" x", np.uint8))
    mats = mats.view("S%d" % n_bit).reshape(mats.shape[:-1])
    print(mats)

    n_bit = 10
    mat = np.random.randint(0, 2**n_bit-1, (n_run, 16), dtype=np.uint16)
    mat[:, n_bit:] = 0
    mat_print = mat[:n_print, :]
    tinybinmat.print(mat_print, n_bit, " x")

    mat16 = test_transpose(mat, n_bit)
    print(mat16[:n_print, :, :])
