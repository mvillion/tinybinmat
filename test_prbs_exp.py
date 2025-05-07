#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np


def prbs_matrix(poly, mat_size):
    lsfr = np.unpackbits(np.array(poly, ">u4").view((np.uint8, 4)))
    lsfr = np.flip(lsfr[-mat_size:])
    mat = np.zeros((mat_size, mat_size), np.uint8)
    mat[:-1, 1:] = np.eye(mat_size-1, dtype=np.uint8)
    mat[:, 0] = lsfr

    mat = np.fliplr(np.rot90(mat))

    return mat


def prbs_1bit(poly, mat_size, out_len, x_status):
    mat = prbs_matrix(poly, mat_size)

    x_status = x_status[:mat_size]

    y = np.empty((out_len, mat_size), dtype=np.uint8)
    for i in range(out_len):
        y[i, :] = x_status
        x_status = (mat @ x_status) % 2
    return y


def mat_all_power(mat1, n_pow):
    mat_all = np.empty_like(mat1, shape=(n_pow,)+mat1.shape)
    mat = mat1
    mat_all[0, :, :] = np.eye(mat1.shape[0], like=mat1)
    for i_pow in range(1, n_pow):
        mat_all[i_pow, :, :] = mat
        mat = (mat1 @ mat) % 2
    return mat, mat_all


def mat_power(mat1, n_pow):
    mat = mat1
    for i_pow in range(1, n_pow):
        mat = (mat1 @ mat) % 2
    return mat


def mat_power_fast(mat1, n_pow):
    mat2 = mat1
    # i_pow2 = 1
    if n_pow & 1:
        mat_acc = mat1
        # i_pow = 1
    else:
        mat_acc = np.eye(mat1.shape[0], like=mat1)
        # i_pow = 0
    for _ in range(1, n_pow.bit_length()):
        mat2 = (mat2 @ mat2) % 2
        # i_pow2 *= 2
        n_pow >>= 1
        if n_pow & 1:
            mat_acc = (mat_acc @ mat2) % 2
            # i_pow += i_pow2
            # print("%s += %s" % (bin(i_pow), bin(i_pow2)))
    # print("i_pow %d" % i_pow)
    return mat_acc


if __name__ == "__main__":

    poly_len = 10
    n_period = (1 << poly_len)-1
    state = prbs_1bit(
        0b1001000000, poly_len, n_period*2, np.ones(poly_len, dtype=np.uint8))
    code = state[:, 0]
    state = state[:1023, :]

    # check code periodicity
    code = code.reshape(2, 1023)
    if np.all(np.diff(code, axis=0) == 0):
        print("code is %d-periodic" % n_period)
    code = code.reshape(-1)[:n_period+poly_len-1]

    # compute matrices equivalent to states
    mat = prbs_matrix(0b1001000000, poly_len)
    _, mat_all = mat_all_power(mat, n_period)
    mat2state = mat_all @ np.ones((n_period, poly_len, 1), dtype=np.uint8)
    mat2state = (mat2state % 2).reshape(n_period, poly_len)

    if np.array_equal(state, mat2state):
        print("matrices do enable to compute PRBS states")
    else:
        raise RuntimeError("oops")

    # compute matrices from states
    i_circul_ax1 = np.arange(n_period).reshape(-1, 1)
    i_circul_ax0 = np.arange(poly_len).reshape(1, -1)
    i_circul = (i_circul_ax1+i_circul_ax0).reshape(-1) % 1023
    circul = state[i_circul, :].reshape(n_period, poly_len, poly_len)
    circul = circul.transpose(0, 2, 1)
    mat_est = circul @ np.repeat(circul[:1, :, :], n_period, axis=0) % 2

    i_bij = (mat_all == mat_est[:1, :, :]).all(axis=(1, 2))

    if np.array_equal(mat_all, mat_est):
        print("matrices do enable to compute PRBS states")
    else:
        raise RuntimeError("oops")

    # check whether matrices are multiple of 3
    mat341 = mat_power(mat_all, 341)
    mat341bis = mat_power_fast(mat_all, 341)
    state341 = mat341 @ np.ones((n_period, poly_len, 1), dtype=np.uint8)
    state341 = (state341 % 2).reshape(n_period, poly_len)

    state16 = np.packbits(state, axis=-1, bitorder="little")
    state16 = state16.view(np.uint16)[:, 0]
