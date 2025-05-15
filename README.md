# tinybinmat
Library for tiny bit matrix manipulation
Bit matrices operate on GF2.
The library is written in C with Python bindings.
8x8, 16x16 and 32x32-bit matrices are supported with optimizations in AVX2.
A 16x16 bit matrix fits into a single AVX2 thus it can be expected that a few operations can be necessary to perform a transposition, multiplication or matrix inversion.
AVX512 proposes _mm512_gf2p8affine_epi64_epi8 instruction that can perform 8 8x8 multiplication in a single instruction.

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

## III.1. bit order

For a 16x16 matrix of 256 bits, chosen order is 16 rows of uint16.
The 1st bit in each row is the LSB of uint16.
Order of bits in row may appear as reversed compared to a decoded matrix of uint8 with 1 bit in each octet but as long as encoding is consistent, a reverse encoding may yield the same results.

Preferred matrix dimension order is C order.
The last dimension is 8, 16 or 32 bit long.
A consequence of this choice is that a single dimension vector can only be encoded as a row vector.
Matrix by vectors operations can only be:
- A*y.T or A*[y_1; y_2; ...; y_N].T
or
- y*A or [y_1; y_2; ...; y_N]*A

If chosen order were 16 columns of uint16, products like A*y with y as a column vector would be possible.
Transposition function converts one choice to the other.

## III.2. supported operations

conversion:
- encode: encode a square matrix into a tinybinmat
- sprint: convert to uint8 array, can be used to reverse encode

transpose:
- transpose: transpose tinybinmat

multiplication:
- mult: multiply twp tinybinmat matrices
- mult_t: multiply a tinybinmat by another transposed tinybinmat
