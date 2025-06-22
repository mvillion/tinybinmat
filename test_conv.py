#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
echo_asm convolution test.

Usage:
    test_tbm [--dummy-exit]

Options:
    -h --help             show this screen
    --dummy-exit          exit for debugger
"""
import tinybinmat as tbm
import numpy as np
import sys


if __name__ == "__main__":
    from docopt import docopt
    arg = docopt(__doc__)
    if arg["--dummy-exit"]:
        print("dummy exit")
        sys.exit()

    aa = np.array([0x2a], np.uint8).reshape(1, -1)
    print(tbm.sprint(tbm.circul(aa, 7), 8, 8, np.arange(2, dtype="u1")))

