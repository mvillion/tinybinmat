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


def test_ok(ok_list, test_str):
    if type(ok_list) is not list:
        if type(ok_list) is not tuple:
            ok_list = (ok_list, None)
        ok_list = [ok_list]
    ok_code = colored("✓", "green")
    ko_code = colored("✗", "red")
    print_list = ["%20s" % test_str]
    for tup in ok_list:
        ok, speed = tup
        print_list.append(ok_code if ok else ko_code)
        if speed is not None:
            print_list.append("%5.1f" % speed)

    print(" ".join(print_list))
    # if not all(ok):
    #     raise RuntimeError("error for %s" % test_str)


def test_encode(mat, n_bit):
    ok_list = []

    decode = tbm.sprint(mat, n_bit, n_bit, np.arange(2, dtype=np.uint8))
    encode = tbm.encode(decode)
    ok_list.append((np.array_equal(mat, encode), None))

    encode = tbm.encode(decode, fmt="gfni")
    decode_gfni = tbm.sprint(
        encode, n_bit, n_bit, np.arange(2, dtype=np.uint8), fmt="gfni")
    ok_list.append((np.array_equal(decode_gfni, decode), None))

    test_ok(ok_list, "encode%d" % (mat.itemsize*8))


def test_mult(mat, matb, n_bit):
    mat8 = tbm.sprint(mat, n_bit, n_bit, np.arange(2, dtype=np.uint8))
    encode = tbm.encode(mat8, fmt="gfni")
    matb8 = tbm.sprint(matb, n_bit, n_bit, np.arange(2, dtype=np.uint8))
    encodeb = tbm.encode(matb8, fmt="gfni")

    t0 = time()
    ref = mat8 @ matb8
    ref &= 1
    ref_duration = time()-t0

    t0 = time()
    prod = tbm.mult(mat, matb)
    duration = time()-t0

    ok_list = []
    for method in ["default", "avx2", "gfni"]:
        t0 = time()
        prod = tbm.mult(mat, matb, method=method)
        duration = time()-t0

        prod = tbm.sprint(prod, n_bit, n_bit, np.arange(2, dtype=np.uint8))
        speed = ref_duration/duration
        ok_list.append((np.array_equal(ref, prod), speed))

    for method in ["gfnio"]:
        t0 = time()
        prod = tbm.mult(encode, encodeb, method=method)
        duration = time()-t0

        prod = tbm.sprint(
            prod, n_bit, n_bit, np.arange(2, dtype=np.uint8), fmt="gfni")
        speed = ref_duration/duration
        ok_list.append((np.array_equal(ref, prod), speed))

    test_ok(ok_list, "mult%d" % (mat.itemsize*8))


def test_mult_t(mat, matb, n_bit):
    mat8 = tbm.sprint(mat, n_bit, n_bit, np.arange(2, dtype=np.uint8))
    encode = tbm.encode(mat8, fmt="gfni")
    matb8 = tbm.sprint(matb, n_bit, n_bit, np.arange(2, dtype=np.uint8))
    encodeb = tbm.encode(matb8, fmt="gfni")

    t0 = time()
    ref = mat8 @ matb8.transpose(0, 2, 1)
    ref &= 1
    ref_duration = time()-t0

    ok_list = []
    for method in ["default", "avx2", "gfni"]:
        t0 = time()
        prod = tbm.mult_t(mat, matb, method=method)
        duration = time()-t0

        prod = tbm.sprint(prod, n_bit, n_bit, np.arange(2, dtype=np.uint8))
        speed = ref_duration/duration
        ok_list.append((np.array_equal(ref, prod), speed))

    for method in ["gfnio"]:
        t0 = time()
        prod = tbm.mult_t(encode, encodeb, method=method)
        duration = time()-t0

        prod = tbm.sprint(
            prod, n_bit, n_bit, np.arange(2, dtype=np.uint8), fmt="gfni")
        speed = ref_duration/duration
        ok_list.append((np.array_equal(ref, prod), speed))

    test_ok(ok_list, "mult_t%d" % (mat.itemsize*8))


def test_transpose(mat, n_bit):
    mat8 = tbm.sprint(mat, n_bit, n_bit, np.arange(2, dtype=np.uint8))
    encode = tbm.encode(mat8, fmt="gfni")

    t0 = time()
    ref = np.ascontiguousarray(mat8.transpose(0, 2, 1))
    ref_duration = time()-t0

    ok_list = []
    for method in ["default", "avx2", "gfni"]:
        t0 = time()
        mat8t = tbm.transpose(mat, method=method)
        duration = time()-t0

        mat8t = tbm.sprint(mat8t, n_bit, n_bit, np.arange(2, dtype=np.uint8))
        speed = ref_duration/duration
        ok_list.append((np.array_equal(ref, mat8t), speed))

    for method in ["gfnio"]:
        t0 = time()
        mat8t = tbm.transpose(encode, method=method)
        duration = time()-t0

        mat8t = tbm.sprint(
            mat8t, n_bit, n_bit, np.arange(2, dtype=np.uint8), fmt="gfni")
        speed = ref_duration/duration
        ok_list.append((np.array_equal(ref, mat8t), speed))

    test_ok(ok_list, "transpose%d" % (mat.itemsize*8))


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

    n_print = 2
    mat_print = mat[:n_print, :]

    mats = tbm.sprint(mat_print, n_bit, n_bit, np.frombuffer(b" x", np.uint8))
    mats = mats.view("S%d" % n_bit).reshape(mats.shape[:-1])
    print(mats)

    for n_bit in [8, 10, 16, 20, 32]:
        n_bit_ceil2 = 2**int(np.ceil(np.log2(n_bit)))
        print("use uint%d for %d" % (n_bit_ceil2, n_bit))
        dtype = np.dtype("uint%d" % n_bit_ceil2)
        mat = np.random.randint(
            0, 2**n_bit-1, (n_run, n_bit_ceil2), dtype=dtype)
        mat[:, n_bit:] = 0
        if n_bit == 6:
            mat_print = mat[:n_print, :]
            tbm.print(mat_print, n_bit, " x")

        test_encode(mat, n_bit)

        mat2 = np.random.randint(
            0, 2**n_bit-1, (n_run, n_bit_ceil2), dtype=dtype)
        mat2[:, n_bit:] = 0

        test_mult(mat, mat2, n_bit)

        test_mult_t(mat, mat2, n_bit)

        test_transpose(mat, n_bit)
