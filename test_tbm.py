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
import tinybinmat as tbm
import numpy as np
import sys

from time import time
from termcolor import colored

method_list = ["default", "avx2", "gfni"]
# method_list = ["avx2", "gfni"]


def test_ok(ok_list, test_str):
    if type(ok_list) is not list:
        ok_list = [ok_list]
    ok_code = colored("✓", "green")
    ko_code = colored("✗", "red")
    print_list = ["%20s" % test_str]
    for tup in ok_list:
        if type(tup) is tuple:
            ok, speed = tup
        else:
            ok, speed = tup, None
        if speed is not None:
            print_list.append("%5.1f" % speed)
        print_list.append(ok_code if ok else ko_code)

    print(" ".join(print_list))
    # if not all(ok):
    #     raise RuntimeError("error for %s" % test_str)


def test_encode(mat, n_bit):
    ok_list = []

    encode = tbm.encode(mat)
    decode_gfni = tbm.sprint(
        encode, n_bit, n_bit, np.arange(2, dtype=np.uint8))
    ok_list.append((np.array_equal(decode_gfni, mat), None))

    test_ok(ok_list, "encode%d" % n_bit)


def test_mult(mat8, matb8, n_bit):
    encode = tbm.encode(mat8)
    encodeb = tbm.encode(matb8)

    t0 = time()
    ref = mat8 @ matb8
    ref &= 1
    ref_duration = time()-t0

    ok_list = []
    for method in method_list:
        try:
            t0 = time()
            prod = tbm.mult(encode, encodeb, method=method)
            duration = time()-t0
        except RuntimeError:
            ok_list.append(False)
            continue

        prod = tbm.sprint(
            prod, n_bit, n_bit, np.arange(2, dtype=np.uint8))
        speed = ref_duration/duration
        ok_list.append((np.array_equal(ref, prod), speed))

    test_ok(ok_list, "mult%d" % n_bit)


def test_mult_t(mat8, matb8, n_bit):
    encode = tbm.encode(mat8)
    encodeb = tbm.encode(matb8)

    t0 = time()
    ref = mat8 @ matb8.transpose(0, 2, 1)
    ref &= 1
    ref_duration = time()-t0

    ok_list = []
    for method in method_list:
        try:
            t0 = time()
            prod = tbm.mult_t(encode, encodeb, method=method)
            duration = time()-t0
        except RuntimeError:
            ok_list.append(False)
            continue

        prod = tbm.sprint(prod, n_bit, n_bit, np.arange(2, dtype=np.uint8))
        speed = ref_duration/duration
        ok_list.append((np.array_equal(ref, prod), speed))

    test_ok(ok_list, "mult_t%d" % n_bit)


def test_transpose(mat8, n_bit):
    encode = tbm.encode(mat8)

    t0 = time()
    ref = np.ascontiguousarray(mat8.transpose(0, 2, 1))
    ref_duration = time()-t0

    ok_list = []
    for method in method_list:
        try:
            t0 = time()
            mat8t = tbm.transpose(encode, method=method)
            duration = time()-t0
        except RuntimeError:
            ok_list.append(False)
            continue

        mat8t = tbm.sprint(mat8t, n_bit, n_bit, np.arange(2, dtype=np.uint8))
        speed = ref_duration/duration
        ok_list.append((np.array_equal(ref, mat8t), speed))

    test_ok(ok_list, "transpose%d" % n_bit)


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

    n_bit = 8
    mat = np.random.randint(0, 2**n_bit-1, (n_run, 8), dtype=np.uint8)
    mat[:, n_bit:] = 0
    mat = mat.view(np.uint64).reshape(n_run, 1, 1)

    n_print = 2
    mat_print = mat[:n_print, :]

    mats = tbm.sprint(mat_print, n_bit, n_bit, np.frombuffer(b" x", np.uint8))
    mats = mats.view("S%d" % n_bit).reshape(mats.shape[:-1])
    print(mats)

    for n_bit in [8, 10, 16, 20, 32]:
        mat = np.random.randint(2, size=(n_run, n_bit, n_bit), dtype=np.uint8)

        test_encode(mat, n_bit)

        mat2 = np.random.randint(2, size=(n_run, n_bit, n_bit), dtype=np.uint8)

        test_mult(mat, mat2, n_bit)

        test_mult_t(mat, mat2, n_bit)

        test_transpose(mat, n_bit)
