# tinybinmat
Library for tiny bit matrix manipulation
Bit matrices operate on GF2.
The library is written in C with Python bindings.

All matrices size are supported with special optimizations for 8x8, 16x16
and 32x32-bit matrices.
AVX2 and GFNI can be used on processors that support these instructions.

# I. installation

Python bindings can be compiled using:

```
>> python3 -m build -w
```
or without virtual-env:
```
>> python3 -m build -w -n
```

# II. usage

```
>> python3 test_tbm
```

# III. details

## III.1. bit order encoding
After initial investigations with 16x16 and 32x32 bit encodings, the dramatic improvement shown using GFNI instruction proved that 8x8 encoding is the best encoding for processors supporting GFNI instructions.
(For other processors 16x16 encoding would likely be more efficient.)

GFNI instructions use little endian order for rows (8 bits in octets) and also for columns.
The 1st octet of the matrix is its last row.
Order of bits is thus:
| row | -: | -: | -: | -: | -: | -: | -: | -: |
| 0 | 63 | 62 | 61 | 60 | 59 | 58 | 57 | 56 |
| 1 | 55 | 54 | 53 | 52 | 51 | 50 | 49 | 48 |
| 2 | 47 | 46 | 45 | 44 | 43 | 42 | 41 | 40 |
| 3 | 39 | 38 | 37 | 36 | 35 | 34 | 33 | 32 |
| 4 | 31 | 30 | 29 | 28 | 27 | 26 | 25 | 24 |
| 5 | 23 | 22 | 21 | 20 | 19 | 18 | 17 | 16 |
| 6 | 15 | 14 | 13 | 12 | 11 | 10 |  9 |  8 |
| 7 |  7 |  6 |  5 |  4 |  3 |  2 |  1 |  0 |

This weird order is likely due to the fact that transposition of the matrix maintains little endian order.

For a matrix of size 10x20, the encoded dimension is 2x3
| A<sub>00</sub> | A<sub>01</sub> | A<sub>02</sub> |
| A<sub>10</sub> | A<sub>11</sub> | A<sub>12</sub> |

Matrix A<sub>02</sub> and A<sub>12</sub> have the last 4 columns with 0.
Matrix A<sub>10</sub> to A<sub>12</sub> have the last 6 rows with 0.


## III.2. supported operations

conversion:
- encode: encode a square matrix into a tinybinmat
- sprint: convert to uint8 array, can be used to reverse encode

transpose:
- transpose: transpose tinybinmat

multiplication:
- mult: multiply twp tinybinmat matrices
- mult_t: multiply a tinybinmat by another transposed tinybinmat

## III.3. AVX(2) instructions

Operations on GF2 imply usage of xor for matrix products and accumulations.

These instructions enable to perform multiple GF2 multiplications or additions at a time:

| instruction                      | n<sub>prod</sub> | n<sub>acc</sub> |
| :------------------------------- | ---: | ---: |
| _mm256_and_si256 (mult)          |  256 |    0 |
| _mm256_xor_si256 (acc)           |    0 |  256 |
| _mm256_gf2p8affineinv_epi64_epi8 | 2048 | 2048 |
| _mm256_popcnt_epi(8,16,32)       |    0 |  256 |

