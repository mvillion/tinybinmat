#include "tinybinmat.h"
#include "tinybinmat_template.c"

#define __SUFFIX(fun) fun##_u64

void tbm_encode_gfnio(
    uint8_t *in, uint64_t n_mat, uint32_t n_row, uint32_t n_col, uint64_t *out)
{
    uint32_t n_octet_col = (n_col+7)/8; //!< number of columns in octets
    uint32_t n_octet_row = (n_row+7)/8; //!< number of rows in octets
    uint8_t n_zero_col = 8*n_octet_col-n_col; //!< number of columns to clear
    uint8_t n_zero_row = 8*n_octet_row-n_row; //!< number of rows to clear

    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        uint8_t *in_mat = in + i_mat*n_row*n_col;
        for (uint8_t i_orow = 0; i_orow < n_octet_row; i_orow++)
        {
            for (uint8_t i_ocol = 0; i_ocol < n_octet_col; i_ocol++)
            {
                uint64_t acc = 0;
                for (uint8_t i_brow = 0; i_brow < 8; i_brow++)
                for (uint8_t i_bcol = 0; i_bcol < 8; i_bcol++)
                {
                    uint64_t i_row = i_brow + i_orow*8;
                    uint64_t i_col = 7-i_bcol + i_ocol*8;
                    uint8_t bit = in_mat[i_row*n_col+i_col] & 1;
                    acc <<= 1;
                    acc |= bit;
                }
                out[(i_mat*n_octet_row+i_orow)*n_octet_col+i_ocol] = acc;
            }
            // clear unused bits in the last columns
            uint64_t zero_mask = 0xff >> n_zero_col;
            zero_mask *= 0x0101010101010101; //!< mask for the last columns
            out[(i_mat*n_octet_row+i_orow+1)*n_octet_col-1] &= zero_mask;
        }
        for (uint8_t i_ocol = 0; i_ocol < n_octet_col; i_ocol++)
        {
            // clear unused bits in the last columns
            uint64_t zero_mask = -1; //!< mask for the last row
            zero_mask <<= n_zero_row*8; 
            out[((i_mat+1)*n_octet_row-1)*n_octet_col+i_ocol] &= zero_mask;
        }
    }
}

void tbm_sprint8_gfnio(
    uint64_t *mat, uint64_t n_mat, uint32_t n_row, uint32_t n_col, char *str01,
    uint8_t *out)
{
    uint32_t n_octet_col = (n_col+7)/8; //!< number of columns in octets
    uint32_t n_octet_row = (n_row+7)/8; //!< number of rows in octets
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        uint8_t *out_mat = out + i_mat*n_row*n_col;
        for (uint8_t i_orow = 0; i_orow < n_octet_row; i_orow++)
        {
            for (uint8_t i_ocol = 0; i_ocol < n_octet_col; i_ocol++)
            {
                uint64_t acc;
                acc = mat[(i_mat*n_octet_row+i_orow)*n_octet_col+i_ocol];
                for (uint8_t i_brow = 0; i_brow < 8; i_brow++)
                for (uint8_t i_bcol = 0; i_bcol < 8; i_bcol++)
                {
                    uint8_t bit = (acc >> 63) & 1;
                    acc <<= 1;
                    uint64_t i_row = i_brow + i_orow*8;
                    uint64_t i_col = 7-i_bcol + i_ocol*8;
                    if (i_row < n_row && i_col < n_col)
                    {
                        out_mat[i_row*n_col+i_col] = str01[bit];
                    }
                }
            }
        }
    }
}

//______________________________________________________________________________
uint64_t inline tbm_transpose8x8_u64(uint64_t in8x8)
{
    // input is 8x8 bit matrix with 8 rows: 0x0001020304050607
    // 1st bit in rows is LSB, thus reversed compared to matrix notation
    uint64_t ur_mask4x4 = 0xf0f0f0f000000000; // up right 4x4 bits
    // uint64_t = dl_mask4x4 = 0x0f0f0f0f00000000; // down left 4x4 bits
    // dl_mask4x4 == ur_mask4x4 << (4*8-4)
    uint64_t xor = in8x8 ^ (in8x8 << 36);
    xor &= ur_mask4x4;
    in8x8 ^= xor;
    xor >>= 36;
    in8x8 ^= xor;
    uint64_t ur_mask2x2 = 0xcccc0000cccc0000; // 4 up right 2x2 bits
    // uint64_t = dl_mask2x2 = 0x3333000033330000; // 4 down left 2x2 bits
    // dl_mask2x2 == ur_mask2x2 << (2*8-2)
    xor = in8x8 ^ (in8x8 << 18);
    xor &= ur_mask2x2;
    in8x8 ^= xor;
    xor >>= 18;
    in8x8 ^= xor;
    uint64_t ur_mask1x1 = 0xaa00aa00aa00aa00; // 16 up right 1x1 bits
    // uint64_t = dl_mask1x1 = 0x5500550055005500; // 16 down left 1x1 bits
    // dl_mask1x1 == ur_mask1x1 << (8-1)
    xor = in8x8 ^ (in8x8 << 9);
    xor &= ur_mask1x1;
    in8x8 ^= xor;
    xor >>= 9;
    in8x8 ^= xor;
    return in8x8;
}

//#define USE_SIMD
#if defined(USE_SIMD)
static void inline tbm_transpose8x8_x2(uint64_t in[2])
{
    for (uint8_t i_prod = 0; i_prod < 2; i_prod++)
        in[i_prod] = tbm_transpose8x8_u64(in[i_prod]);
}

static void inline tbm_transpose8x8_x4(uint64_t in[4])
{
    for (uint8_t i_prod = 0; i_prod < 4; i_prod += 2)
        tbm_transpose8x8_x2(in+i_prod);
}
#else
static void inline tbm_transpose8x8_x4(uint64_t in[4])
{
    for (uint8_t i_prod = 0; i_prod < 4; i_prod++)
        in[i_prod] = tbm_transpose8x8_u64(in[i_prod]);
}
#endif

static __attribute__ ((noinline)) void tbm_transpose_1d(
    uint64_t *in, uint64_t n8x8, uint64_t *out)
{
    for (uint64_t i8x8 = 0; i8x8 < n8x8; i8x8++)
        out[i8x8] = tbm_transpose8x8_u64(in[i8x8]);
}

#pragma GCC push_options //----------------------------------------------------
#pragma GCC optimize("no-tree-vectorize")
static void __attribute__ ((noinline)) tbm_transpose_256(
    uint64_t *in, uint64_t n_mat, uint32_t n_row8, uint32_t n_col8, 
    uint64_t *out)
{
    tbm_transpose_256_template(
        in, n_mat, n_row8, n_col8, out, tbm_transpose8x8_x4);
}

void __SUFFIX(tbm_transpose) (
    uint64_t *in, uint64_t n_mat, uint32_t n_row8, uint32_t n_col8, 
    uint64_t *out)
{
    if (n_row8 == 1 || n_col8 == 1)
    {
        // no block transpose needed
        uint64_t n8x8 = n_mat*n_row8*n_col8; //!< number of 8x8 blocks
        return tbm_transpose_1d(in, n8x8, out);
    }
    // else if (n_row8 == 2 && n_col8 == 2)
    // {
    //     return tbm_transpose_2x2((__m256i *)in, n_mat, (__m256i *)out);
    // }
    // else if (n_row8 == 4 && n_col8 == 4)
    // {
    //     return tbm_transpose_4x4((__m256i *)in, n_mat, (__m256i *)out);
    // }
    tbm_transpose_256(in, n_mat, n_row8, n_col8, out);
}
#pragma GCC pop_options //-----------------------------------------------------

//______________________________________________________________________________
// multiply two 8x8 bit matrices
uint64_t inline tbm_mult8x8_u64(uint64_t a, uint64_t b)
{
    uint64_t out = 0;
    uint8_t *b8 = (uint8_t *)&b;
    for (uint8_t i_bit = 0; i_bit < 8; i_bit++)
    {
        // create bit mask from the least significant bits in a
        uint64_t bit_a = a & 0x0101010101010101;
        bit_a *= 0xff;
        a >>= 1;
        uint64_t prod = bit_a & (0x0101010101010101*b8[7-i_bit]);
        out ^= prod;
    }
    return out;
}

static void inline tbm_mult8x8_1x4(
    uint64_t a, uint64_t b[4], uint64_t out[4])
{
    for (uint8_t i_prod = 0; i_prod < 4; i_prod++)
        out[i_prod] = tbm_mult8x8_u64(a, b[i_prod]);
}

static void inline tbm_mult8x8_x4(
    uint64_t a[4], uint64_t b[4], uint64_t out[4])
{
    for (uint8_t i_prod = 0; i_prod < 4; i_prod++)
        out[i_prod] = tbm_mult8x8_u64(a[i_prod], b[i_prod]);
}

static void __attribute__ ((noinline)) tbm_mult_256(
    uint64_t *in, uint64_t n_mat, uint32_t n_row8, uint32_t n_col8,
    uint64_t *in2, uint32_t n_col8_2, uint64_t *out)
{
    tbm_mult_256_template(
        in, n_mat, n_row8, n_col8, in2, n_col8_2, out, tbm_mult8x8_1x4);
}

static void __attribute__ ((noinline)) tbm_mult_ncol8_1(
    uint64_t *in, uint64_t n_mat, uint64_t *in2, uint64_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
        out[i_mat] = tbm_mult8x8_u64(in[i_mat], in2[i_mat]);
}

static void __attribute__ ((noinline)) tbm_mult_ncol8_2(
    uint64_t *in, uint64_t n_mat, uint64_t *in2, uint64_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        out[0] = tbm_mult8x8_u64(in[0], in2[0]);
        out[0] ^= tbm_mult8x8_u64(in[1], in2[2]);
        out[1] = tbm_mult8x8_u64(in[0], in2[1]);
        out[1] ^= tbm_mult8x8_u64(in[1], in2[3]);
        out[2] = tbm_mult8x8_u64(in[2], in2[0]);
        out[2] ^= tbm_mult8x8_u64(in[3], in2[2]);
        out[3] = tbm_mult8x8_u64(in[2], in2[1]);
        out[3] ^= tbm_mult8x8_u64(in[3], in2[3]);
        in += 4;
        in2 += 4;
        out += 4;
    }
}

static void __attribute__ ((noinline)) tbm_mult_ncol8_4(
    uint64_t *in, uint64_t n_mat, uint64_t *in2, uint64_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        tbm_mult32x32_template(in, in2, out, tbm_mult8x8_1x4);
        in += 16;
        in2 += 16;
        out += 16;
    }
}

void __SUFFIX(tbm_mult) (
    uint64_t *in, uint64_t n_mat, uint32_t n_row8, uint32_t n_col8,
    uint64_t *in2, uint32_t n_col8_2, uint64_t *out)
{
    if ((n_col8 == 1) && (n_row8 == 1) && (n_col8_2 == 1))
        return tbm_mult_ncol8_1(in, n_mat, in2, out);
    if ((n_col8 == 2) && (n_row8 == 2) && (n_col8_2 == 2))
        return tbm_mult_ncol8_2(in, n_mat, in2, out);
    if ((n_col8 == 4) && (n_row8 == 4) && (n_col8_2 == 4))
        return tbm_mult_ncol8_4(in, n_mat, in2, out);
    
    tbm_mult_256(in, n_mat, n_row8, n_col8, in2, n_col8_2, out);
}

//______________________________________________________________________________
// multiply two 8x8 bit matrices with the second matrix transposed
uint64_t inline tbm_mult_t8x8_u64(uint64_t a8x8, uint64_t tb8x8)
{
    uint64_t out = 0;
    uint8_t *tb = (uint8_t *)&tb8x8;
    for (uint8_t i_bit = 0; i_bit < 8; i_bit++)
    {
        uint64_t repeat = 0x0101010101010101*tb[7-i_bit];
        uint64_t prod = a8x8 & repeat;
        prod ^= prod << 4;
        prod ^= prod << 2;
        prod ^= prod << 1;
        prod &= 0x8080808080808080;
        out >>= 1;
        out |= prod;
    }
    return out;
}

uint64_t inline tbm_dot_t(uint64_t a[4], uint64_t b[4], uint32_t n_dot)
{
    uint64_t out = 0;
    for (uint8_t i_dot = 0; i_dot < n_dot; i_dot++)
        out ^= tbm_mult_t8x8_u64(a[i_dot], b[i_dot]);
    return out;
}

static void __attribute__ ((noinline)) tbm_mult_t_ncol8_1(
    uint64_t *in, uint64_t n_mat, uint64_t *in2t, uint64_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
        out[i_mat] = tbm_mult_t8x8_u64(in[i_mat], in2t[i_mat]);
}

#pragma GCC push_options //-----------------------------------------------------
#pragma GCC optimize("no-tree-vectorize")
static void __attribute__ ((noinline)) tbm_mult_t_dot(
    uint64_t *in, uint64_t n_mat, uint32_t n_row8, uint32_t n_col8,
    uint64_t *in2, uint32_t n_col8_2, uint64_t *out)
{
    tbm_mult_t_dot_template(
        in, n_mat, n_row8, n_col8, in2, n_col8_2, out, tbm_dot_t);
}

#pragma GCC pop_options //------------------------------------------------------

void __SUFFIX(tbm_mult_t) (
    uint64_t *in, uint64_t n_mat, uint32_t n_row8, uint32_t n_col8,
    uint64_t *in2, uint32_t n_row8_2, uint64_t *out)
{
    if ((n_col8 == 1) && (n_row8 == 1) && (n_row8_2 == 1))
        return tbm_mult_t_ncol8_1(in, n_mat, in2, out);
    // if ((n_col8 == 2) && (n_row8 == 2) && (n_row8_2 == 2))
    //     return tbm_mult_t_ncol8_2(in, n_mat, in2, out);
    // if ((n_col8 == 4) && (n_row8 == 4) && (n_row8_2 == 4))
    //     return tbm_mult_t_ncol8_4(in, n_mat, in2, out);
    
    tbm_mult_t_dot(in, n_mat, n_row8, n_col8, in2, n_row8_2, out);
}
