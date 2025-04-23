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

from matplotlib import pyplot as plt
from termcolor import colored
from termcolor import cprint
from time import time

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

    n_run = 2
    n_bit = 10
    mat = np.random.randint(0, 2**n_bit-1, (n_run, 16), dtype=np.uint16)
    mat = mat.copy()
    tinybinmat.print(mat, n_bit, " x")

