#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

from time import time

code_length = 1023

global_mat = {}

l1ca_init = np.array([
    0x0e53f,
    0x1f75b,
    0x3d393,
    0x79a03,
    0x7e2ed,
    0xff8ff,
    0x3ebed,
    0x7eaff,
    0xfe8db,
    0x004c7,
    0x034ab,
    0x095c3,
    0x116a3,
    0x21063,
    0x41de3,
    0x806e3,
    0x02c9d,
    0x0641f,
    0x0f51b,
    0x1d713,
    0x39303,
    0x71b23,
    0x00cd5,
    0x0d553,
    0x19783,
    0x31223,
    0x61963,
    0xc0fe3,
    0x0ed2d,
    0x1e77f,
    0x3f3db,
    0x7da93,
], np.uint32)


def l1ca_shift(num):
    g1g2_poly = 0xe8492  # 0b1110_1000_0100_1001_0010|1
    current = num & 1
    return (num >> 1) ^ current*g1g2_poly


def make_l1ca(prn, prn_len):
    x = l1ca_init[prn]
    y = np.zeros(prn_len, dtype=bool)
    for i in range(prn_len-1):
        y[i] = x & 1
        x = l1ca_shift(x)
    y[prn_len-1] = x & 1
    return y, x


# code using galois field division
def gf_div(poly, out_len, x_status):
    # print("poly {:020b}".format(poly))
    y = np.zeros(out_len, dtype=bool)
    for i in range(out_len):
        current = x_status & 1
        y[i] = current
        # if i < 10:
        #     print("x_status {:020b}".format(x_status))
        x_status = (x_status >> 1) ^ current*poly
    return y


def l1ca_gf_matrix(poly, mat_size):
    lsfr = np.unpackbits(np.array(poly, ">u4").view((np.uint8, 4)))
    lsfr = np.flip(lsfr[-mat_size:])
    mat = np.zeros((mat_size, mat_size), np.uint8)
    mat[:-1, 1:] = np.eye(mat_size-1, dtype=np.uint8)
    mat[:, 0] = lsfr
    return mat


def l1ca_gf_1bit(poly, out_len, x_status):
    mat_size = 2*10  # (2**10-1)
    mat = l1ca_gf_matrix(poly, mat_size)

    x_status = np.unpackbits(np.array(x_status, ">u4").view((np.uint8, 4)))
    x_status = np.flip(x_status[-mat_size:])

    y = np.zeros(out_len, dtype=bool)
    for i in range(out_len):
        y[i] = x_status[0]
        # if i < 10:
        #     print("x_status %s" % "".join(reversed([str(k) for k in x_status])))
        x_status = (mat @ x_status) % 2
    return y


def l1ca_prbs_1bit(poly, out_len, x_status):
    mat_size = 2*10  # (2**10-1)
    mat = l1ca_gf_matrix(poly, mat_size)

    mat = np.fliplr(np.rot90(mat))

    x_status = x_status[:mat_size]

    y = np.zeros(out_len, dtype=bool)
    for i in range(out_len):
        y[i] = x_status[0]
        x_status = (mat @ x_status) % 2
    return y


def l1ca_prbs_20bit(poly, out_len, x_status):
    mat_size = 2*10  # (2**10-1)

    try:
        mat20 = global_mat["mat20"]
    except KeyError:
        mat = l1ca_gf_matrix(poly, mat_size)

        mat = np.fliplr(np.rot90(mat))

        mat2 = (mat @ mat) % 2
        mat4 = (mat2 @ mat2) % 2
        mat8 = (mat4 @ mat4) % 2
        mat16 = (mat8 @ mat8) % 2
        mat20 = (mat16 @ mat4) % 2

        global_mat["mat20"] = mat20

    x_status = x_status[:mat_size]

    n_step = out_len
    n_step += mat_size-1
    n_step //= mat_size
    y = np.zeros((n_step, mat_size), dtype=bool)
    for i in range(n_step):
        y[i, :] = x_status
        x_status = (mat20 @ x_status) % 2
    return y.reshape(-1)[:out_len]


def l1ca_prbs_32bit(poly, out_len, x_status):
    mat_size = 2*10  # (2**10-1)
    ext_size = 32

    try:
        mat32 = global_mat["mat32"]
    except KeyError:
        mat = l1ca_gf_matrix(poly, mat_size)

        mat_20x20 = np.fliplr(np.rot90(mat))

        n_ext = ext_size-mat_size
        mat = np.zeros((ext_size, ext_size), np.uint32)
        mat[-mat_size:, -mat_size:] = mat_20x20
        mat[:n_ext, 1:n_ext+1] = np.eye(n_ext, dtype=np.uint8)

        mat2 = (mat @ mat) % 2
        mat4 = (mat2 @ mat2) % 2
        mat8 = (mat4 @ mat4) % 2
        mat16 = (mat8 @ mat8) % 2
        mat32 = (mat16 @ mat16) % 2

        global_mat["mat32"] = mat32

    n_step = out_len
    n_step += ext_size-1
    n_step //= ext_size
    y = np.zeros((n_step, ext_size), dtype=bool)
    for i in range(n_step):
        y[i, :] = x_status
        x_status = (mat32 @ x_status) % 2
    return y.reshape(-1)[:out_len]


def l1ca_tiny_32bit(poly, out_len, x_status):
    mat_size = 2*10  # (2**10-1)
    ext_size = 32
    import tinybinmat as tbm

    try:
        tiny32 = global_mat["tiny32"]
    except KeyError:
        mat = l1ca_gf_matrix(poly, mat_size)

        mat_20x20 = np.fliplr(np.rot90(mat))

        n_ext = ext_size-mat_size
        mat = np.zeros((ext_size, ext_size), np.uint32)
        mat[-mat_size:, -mat_size:] = mat_20x20
        mat[:n_ext, 1:n_ext+1] = np.eye(n_ext, dtype=np.uint8)

        tiny = tbm.encode(mat.astype(np.uint8))

        tiny32 = tiny
        for i_pow in range(5):
            tiny32 = tbm.mult(tiny32, tiny32)

        global_mat["tiny32"] = tiny32

    # use matrix for x_status as vector is not supported yet
    x_status_square = np.zeros((ext_size, ext_size), np.uint8)
    x_status_square[:, 0] = x_status
    x_status_square = tbm.encode(x_status_square)
    n_step = out_len
    n_step += ext_size-1
    n_step //= ext_size
    y = np.zeros((n_step, ext_size), dtype=bool)
    for i in range(n_step):
        y[i] = x_status_square
        x_status_square = tbm.mult(tiny32, x_status_square)
    return y.reshape(-1)[:out_len]


if __name__ == "__main__":
    code_ref = []
    for prn in range(32):
        code, end_state = make_l1ca(prn, code_length)

        code_ref.append(code)
        # code_print = [str(int(k)) for k in code_ref[:64]]
        # print("code_ref %s" % "".join(code_print))

    test_fun_list = [
        gf_div, l1ca_gf_1bit, l1ca_prbs_1bit, l1ca_prbs_20bit,
        l1ca_prbs_32bit, l1ca_tiny_32bit]

    for test_fun in test_fun_list:
        t0 = time()
        for prn in range(32):
            if "gf" in test_fun.__name__:
                seed = l1ca_init[prn]
            else:
                seed = code_ref[prn][:32]
            code = test_fun(0xe8492, code_length, seed)
            if (code == code_ref[prn]).all():
                print(".", end="")
            else:
                raise RuntimeError("error in prn %d" % (prn+1))
        duration = time()-t0
        print("\n%s: %f" % (test_fun.__name__, duration))
