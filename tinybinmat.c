#include "tinybinmat.h"
#include "tinybinmat_template.c"
#include <stdio.h>

#if defined(USE_SIMD)
#include "arm_neon.h"
#define __SUFFIX(fun) fun##_simd

void print_simd_uint64(uint64x2_t reg)
{
    uint64_t *ptr = (uint64_t *)&reg;
    printf("%016lx ", ptr[1]);
    printf("%016lx\n", ptr[0]);
}
#else
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
#endif

//______________________________________________________________________________
uint64_t inline tbm_transpose8x8_u64(uint64_t in8x8)
{// input is 8x8 bit matrix with 8 rows: 0x0001020304050607
    // 1st bit in rows is LSB, thus reversed compared to matrix notation
    uint64_t ur_mask4x4 = 0xf0f0f0f000000000; // up right 4x4 bits
    // uint64_t = ur_mask4x4 = 0xf0f0f0f000000000;
    //            dl_mask4x4 >> (4*8+4) // down left 4x4 bits
    //            dl_mask4x4 = 0x000000000f0f0f0f
    // fedcba9876543210
    // f0f0f0f000000000
    // f d b 9
    //          f d b 9
    //          6 4 2 0
    // 6 4 2 0
    // 6e4c2a087f5d3b19
    //----------------
    uint64_t xor = in8x8 ^ (in8x8 << 36);
    xor &= ur_mask4x4;
    in8x8 ^= xor;
    xor >>= 36;
    in8x8 ^= xor;
    uint64_t ur_mask2x2 = 0xcccc0000cccc0000; // 4 up right 2x2 bits
    // uint64_t = dl_mask2x2 = 0x0000333300003333; // 4 down left 2x2 bits
    // dl_mask2x2 == ur_mask2x2 >> (2*8+2)
    xor = in8x8 ^ (in8x8 << 18);
    xor &= ur_mask2x2;
    in8x8 ^= xor;
    xor >>= 18;
    in8x8 ^= xor;
    uint64_t ur_mask1x1 = 0xaa00aa00aa00aa00; // 16 up right 1x1 bits
    // uint64_t = dl_mask1x1 = 0x0055005500550055; // 16 down left 1x1 bits
    // dl_mask1x1 == ur_mask1x1 >> (8+1)
    xor = in8x8 ^ (in8x8 << 9);
    xor &= ur_mask1x1;
    in8x8 ^= xor;
    xor >>= 9;
    in8x8 ^= xor;
    return in8x8;
}

#if defined(USE_SIMD)
static void inline tbm_transpose8x8_simd(uint64_t in[2])
{
    uint64x2_t in8x8 = vld1q_u64(in);
    uint64x2_t xor;
#if 0
    uint64x2_t ur_mask4x4 = vdupq_n_u64(0xf0f0f0f000000000);
    xor = veorq_u64(in8x8, vshlq_n_u64(in8x8, 36));
    xor = vandq_u64(xor, ur_mask4x4);
    in8x8 = veorq_u64(in8x8, xor);
    xor = vshrq_n_u64(xor, 36);
    in8x8 = veorq_u64(in8x8, xor);
#else
    // we want:
    // 6e4c2a087f5d3b19
    // two options are possible:
    // .6.4.2.0.7.5.3.1 (bof shli(x, x >> 4, 32)) and:
    // .e.c.a.8.f.d.b.9 (ok shri(x, x, 36))
    // or
    // 6.4.2.0.7.5.3.1. (ok shli(x, x, 36)) and:
    // e.c.a.8.f.d.b.9. bof shri(x, x << 4, 32))
    uint64x2_t in_6420_7531 = vsliq_n_u64(vshrq_n_u64(in8x8, 4), in8x8, 32);
    uint64x2_t in_eca8_fdb9 = vsriq_n_u64(in8x8, in8x8, 36);
    in8x8 = vreinterpretq_u64_u8(vsliq_n_u8(
        vreinterpretq_u8_u64(in_eca8_fdb9),
        vreinterpretq_u8_u64(in_6420_7531), 4));
#endif
    uint64x2_t ur_mask2x2 = vdupq_n_u64(0xcccc0000cccc0000);
    xor = veorq_u64(in8x8, vshlq_n_u64(in8x8, 18));
    xor = vandq_u64(xor, ur_mask2x2);
    in8x8 = veorq_u64(in8x8, xor);
    xor = vshrq_n_u64(xor, 18);
    in8x8 = veorq_u64(in8x8, xor);
    uint64x2_t ur_mask1x1 = vdupq_n_u64(0xaa00aa00aa00aa00);
    xor = veorq_u64(in8x8, vshlq_n_u64(in8x8, 9));
    xor = vandq_u64(xor, ur_mask1x1);
    in8x8 = veorq_u64(in8x8, xor);
    xor = vshrq_n_u64(xor, 9);
    in8x8 = veorq_u64(in8x8, xor);
    vst1q_u64(in, in8x8);
}

static void inline tbm_transpose8x8_x4(uint64_t in[4])
{
    for (uint8_t i_prod = 0; i_prod < 4; i_prod += 2)
        tbm_transpose8x8_simd(in+i_prod);
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
    uint64_t i8x8; //!< index for 4 8x8 blocks
    for (i8x8 = 0; i8x8 < n8x8/4*4; i8x8 += 4)
    {
        for (uint8_t i_copy = 0; i_copy < 4; i_copy++)
            out[i_copy] = in[i_copy];
        tbm_transpose8x8_x4(out);
        in += 4;
        out += 4;
    }
    if (i8x8 == n8x8)
        return; // all blocks are processed
    uint64_t tmp[4];
        for (uint8_t i_copy = 0; i_copy < 4; i_copy++)
            tmp[i_copy] = in[i_copy];
    tbm_transpose8x8_x4(tmp);
    for (i8x8 = 0; i8x8 < (n8x8 & 3); i8x8++)
        out[i8x8] = tmp[i8x8];
}

static __attribute__ ((noinline)) void tbm_transpose_2x2(
    uint64_t *in, uint64_t n8x8, uint64_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n8x8; i_mat++)
    {
#if defined(USE_SIMD)
        uint64x2x2_t out3_0 = vld2q_u64(in);
        vst1q_u64(out, out3_0.val[0]);
        vst1q_u64(out+2, out3_0.val[1]);
#else
        out[0] = in[0];
        out[1] = in[2];
        out[2] = in[1];
        out[3] = in[3];
#endif
        tbm_transpose8x8_x4(out);
        in += 4;
        out += 4;
    }
}

static __attribute__ ((noinline)) void tbm_transpose_4x4(
    uint64_t *in, uint64_t n8x8, uint64_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n8x8; i_mat++)
    {
#if defined(USE_SIMD)
        uint64x2x4_t out7_0 = vld4q_u64(in);
        uint64x2x4_t out15_8 = vld4q_u64(in+8);
        for (uint8_t i_4 = 0; i_4 < 4; i_4++)
        {
            vst1q_u64(out+i_4*4, out7_0.val[i_4]);
            vst1q_u64(out+i_4*4+2, out15_8.val[i_4]);
        }
#else
        out[ 0] = in[ 0+0];
        out[ 1] = in[ 4+0];
        out[ 2] = in[ 8+0];
        out[ 3] = in[12+0];
        out[ 4] = in[ 0+1];
        out[ 5] = in[ 4+1];
        out[ 6] = in[ 8+1];
        out[ 7] = in[12+1];
        out[ 8] = in[ 0+2];
        out[ 9] = in[ 4+2];
        out[10] = in[ 8+2];
        out[11] = in[12+2];
        out[12] = in[ 0+3];
        out[13] = in[ 4+3];
        out[14] = in[ 8+3];
        out[15] = in[12+3];
#endif
        tbm_transpose8x8_x4(out);
        tbm_transpose8x8_x4(out+4);
        tbm_transpose8x8_x4(out+8);
        tbm_transpose8x8_x4(out+12);
        in += 16;
        out += 16;
    }
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
        return tbm_transpose_1d(in, n_mat, out);
    }
    else if (n_row8 == 2 && n_col8 == 2)
    {
        return tbm_transpose_2x2(in, n_mat, out);
    }
    else if (n_row8 == 4 && n_col8 == 4)
    {
        return tbm_transpose_4x4(in, n_mat, out);
    }
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

#if defined(USE_SIMD)
uint64_t inline tbm_mult8x8_u64rev(uint64_t a, uint64_t b)
{
    uint64_t out = 0;
    uint8_t *b8 = (uint8_t *)&b;
    for (uint8_t i_bit = 0; i_bit < 8; i_bit++)
    {
        // create bit mask from the least significant bits in a
        uint64_t bit_a = a & 0x8080808080808080;
        bit_a >>= 7;
        bit_a *= 0xff;
        a <<= 1;
        uint64_t prod = bit_a & (0x0101010101010101*b8[i_bit]);
        out ^= prod;
    }
    return out;
}

static uint8x16_t inline tbm_mult8x8_simd(uint8x16_t a, uint8x16_t b)
{
    uint8x16_t repeat8x2 = vdupq_n_u8(0);
    repeat8x2 = vreinterpretq_u8_u64(vsetq_lane_u64(
        0x0808080808080808ull, vreinterpretq_u64_u8(repeat8x2), 1));
    uint8x16_t out = vdupq_n_u8(0);
    for (uint8_t i_bit = 0; i_bit < 8; i_bit++)
    {
        // create bit mask from the most significant bits in a octets
        uint8x16_t bit_a = vreinterpretq_u8_s8(
            vshrq_n_s8(vreinterpretq_s8_u8(a), 7));
        a = vshlq_n_u8(a, 1);
        // repeat 8 times least significant octet of b 8x8 matrices
        uint8x16_t b_repeat = vqtbl1q_u8(b, repeat8x2);
        b = vreinterpretq_u8_u64(vshrq_n_u64(vreinterpretq_u64_u8(b), 8));
        uint8x16_t prod = vandq_u8(bit_a, b_repeat);
        out = veorq_u8(out, prod);
    }
    return out;
}

static void inline tbm_mult8x8_1x4(uint64_t a, uint64_t b[4], uint64_t out[4])
{
    uint8x16_t a_x2 = vreinterpretq_u8_u64(vdupq_n_u64(a));
    uint8x16_t b_x2 = vld1q_u8((uint8_t *)b);
    vst1q_u8((uint8_t *)out, tbm_mult8x8_simd(a_x2, b_x2));
    b_x2 = vld1q_u8((uint8_t *)(b+2));
    vst1q_u8((uint8_t *)(out+2), tbm_mult8x8_simd(a_x2, b_x2));
}

static void inline tbm_mult8x8_x4(uint64_t a[4], uint64_t b[4], uint64_t out[4])
{
    uint8x16_t a_x2 = vld1q_u8((uint8_t *)a);
    uint8x16_t b_x2 = vld1q_u8((uint8_t *)b);
    vst1q_u8((uint8_t *)out, tbm_mult8x8_simd(a_x2, b_x2));
    a_x2 = vld1q_u8((uint8_t *)(a+2));
    b_x2 = vld1q_u8((uint8_t *)(b+2));
    vst1q_u8((uint8_t *)(out+2), tbm_mult8x8_simd(a_x2, b_x2));
}

static void inline tbm_mult16x16_simd(
    uint64_t a[4], uint64_t b[4], uint64_t out[4])
{
    // a is:    b is:
    // [0, 1]   [0, 1]   [0, 0]   [0, 1]   [1, 1]   [2, 3]
    // [2, 3] @ [2, 3] = [2, 2] * [0, 1] + [3, 3] * [2, 3]
    uint8x16_t b_01 = vld1q_u8((uint8_t *)b);
    uint8x16_t b_23 = vld1q_u8((uint8_t *)(b+2));
    uint8x16_t acc;
    acc = tbm_mult8x8_simd(vreinterpretq_u8_u64(vdupq_n_u64(a[0])), b_01);
    acc = veorq_u8(acc, tbm_mult8x8_simd(
        vreinterpretq_u8_u64(vdupq_n_u64(a[1])), b_23));
    vst1q_u8((uint8_t *)out, acc);
    acc = tbm_mult8x8_simd(vreinterpretq_u8_u64(vdupq_n_u64(a[2])), b_01);
    acc = veorq_u8(acc, tbm_mult8x8_simd(
        vreinterpretq_u8_u64(vdupq_n_u64(a[3])), b_23));
    vst1q_u8((uint8_t *)(out+2), acc);
}
#else
static void inline tbm_mult8x8_1x4(uint64_t a, uint64_t b[4], uint64_t out[4])
{
    for (uint8_t i_prod = 0; i_prod < 4; i_prod++)
        out[i_prod] = tbm_mult8x8_u64(a, b[i_prod]);
}

static void inline tbm_mult8x8_x4(uint64_t a[4], uint64_t b[4], uint64_t out[4])
{
    for (uint8_t i_prod = 0; i_prod < 4; i_prod++)
        out[i_prod] = tbm_mult8x8_u64(a[i_prod], b[i_prod]);
}
#endif

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
    uint64_t i8x8; //!< index for 4 8x8 blocks
    for (i8x8 = 0; i8x8 < n_mat/4*4; i8x8 += 4)
    {
        tbm_mult8x8_x4(in, in2, out);
        in += 4;
        in2 += 4;
        out += 4;
    }
    if (i8x8 == n_mat)
        return; // all blocks are processed
    uint64_t tmp[4];
    tbm_mult8x8_x4(in, in2, tmp);
    for (i8x8 = 0; i8x8 < (n_mat & 3); i8x8++)
        out[i8x8] = tmp[i8x8];
}

static void __attribute__ ((noinline)) tbm_mult_ncol8_2(
    uint64_t *in, uint64_t n_mat, uint64_t *in2, uint64_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
#if defined(USE_SIMD)
        tbm_mult16x16_simd(in, in2, out);
#else
        out[0] = tbm_mult8x8_u64(in[0], in2[0]);
        out[0] ^= tbm_mult8x8_u64(in[1], in2[2]);
        out[1] = tbm_mult8x8_u64(in[0], in2[1]);
        out[1] ^= tbm_mult8x8_u64(in[1], in2[3]);
        out[2] = tbm_mult8x8_u64(in[2], in2[0]);
        out[2] ^= tbm_mult8x8_u64(in[3], in2[2]);
        out[3] = tbm_mult8x8_u64(in[2], in2[1]);
        out[3] ^= tbm_mult8x8_u64(in[3], in2[3]);
#endif
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
//#define USE_SIMD
#if defined(USE_SIMD)
uint64_t inline tbm_mult_t8x8_u64(uint64_t a8x8, uint64_t tb8x8)
{
    uint8x8_t acc = vcreate_u8(0);
    uint8x8_t _a8x8 = vcreate_u8(a8x8);
    uint8x8_t _tb8x8 = vcreate_u8(tb8x8);
    for (uint8_t i_bit = 0; i_bit < 8; i_bit++)
    {
        // uint8x8_t repeat = vdup_n_u8(tb[i_bit]);
        uint8x8_t repeat = vdup_lane_u8(_tb8x8, i_bit);
        uint8x8_t prod = vand_u8(_a8x8, repeat);
        prod = vcnt_u8(prod);
        acc = vsli_n_u8(prod, acc, 1);
    }
    return vdupd_lane_u64(vreinterpret_u64_u8(acc), 0);
}

#if defined(USE_DOT) || 1
uint8x16_t inline tbm_mult_t8x8_simd(uint8x16_t a8x8_x2, uint8x16_t tb8x8_x2)
{
    uint8x16_t repeat8x2 = vcombine_u8(vdup_n_u8(0), vdup_n_u8(8));
    uint8x16_t acc_x2 = vdupq_n_u8(0);
    #pragma GCC unroll 8
    for (uint8_t i_bit = 0; i_bit < 8; i_bit++)
    {
        // uint8x16_t repeat = vcombine_u8(
        //     vdup_lane_u8(vget_low_u8(tb8x8_x2), i_bit),
        //     vdup_lane_u8(vget_high_u8(tb8x8_x2), i_bit));
        uint8x16_t repeat = vqtbl1q_u8(tb8x8_x2, repeat8x2);
        tb8x8_x2 = vreinterpretq_u8_u64(
            vshrq_n_u64(vreinterpretq_u64_u8(tb8x8_x2), 8));
        uint8x16_t prod = vandq_u8(a8x8_x2, repeat);
        prod = vcntq_u8(prod);
        acc_x2 = vsliq_n_u8(prod, acc_x2, 1);
    }
    return acc_x2;
}
#else
// non-functional!
uint8x16_t inline tbm_mult_t8x8_simd(uint8x16_t a8x8_x2, uint8x16_t tb8x8_x2)
{
    uint8x16_t test_bit = vdupq_n_u8(0);
    uint8x16_t acc_x2 = vdupq_n_u8(0);
    #pragma GCC unroll 8
    for (uint8_t i_bit = 0; i_bit < 8; i_bit++)
    {
        uint8x16_t prod = vandq_u8(
            vtstq_u8(a8x8_x2, test_bit), vtstq_u8(tb8x8_x2, test_bit));
        test_bit = vshlq_n_u8(test_bit, 1);
        acc_x2 = vorrq_u8(acc_x2, prod);
    }
    return acc_x2;
}
#endif

static void inline tbm_mult_t8x8_x4(
    uint64_t a[4], uint64_t b[4], uint64_t out[4])
{
    uint8x16_t a_x2 = vld1q_u8((uint8_t *)a);
    uint8x16_t b_x2 = vld1q_u8((uint8_t *)b);
    vst1q_u8((uint8_t *)out, tbm_mult_t8x8_simd(a_x2, b_x2));
    a_x2 = vld1q_u8((uint8_t *)(a+2));
    b_x2 = vld1q_u8((uint8_t *)(b+2));
    vst1q_u8((uint8_t *)(out+2), tbm_mult_t8x8_simd(a_x2, b_x2));
}
#else
uint64_t inline tbm_mult_t8x8_u64(uint64_t a8x8, uint64_t tb8x8)
{
    uint64_t acc = 0;
    uint8_t *tb = (uint8_t *)&tb8x8;
    for (uint8_t i_bit = 0; i_bit < 8; i_bit++)
    {
        uint64_t repeat = 0x0101010101010101*tb[i_bit];
        uint64_t prod = a8x8 & repeat;
        prod ^= prod >> 4;
        prod ^= prod >> 2;
        prod ^= prod >> 1;
        prod &= 0x0101010101010101;
        acc <<= 1;
        acc |= prod;
    }
    return acc;
}

static void inline tbm_mult_t8x8_x4(uint64_t a[4], uint64_t b[4], uint64_t out[4])
{
    for (uint8_t i_prod = 0; i_prod < 4; i_prod++)
        out[i_prod] = tbm_mult_t8x8_u64(a[i_prod], b[i_prod]);
}
#endif

uint64_t inline tbm_dot_t(uint64_t a[4], uint64_t b[4], uint32_t n_dot)
{
    uint64_t out = 0;
    for (uint8_t i_dot = 0; i_dot < n_dot; i_dot++)
        out ^= tbm_mult_t8x8_u64(a[i_dot], b[i_dot]);
    return out;
}

static void __attribute__ ((noinline)) tbm_mult_t_ncol8_1(
    uint64_t *in, uint64_t n_mat, uint64_t *in2, uint64_t *out)
{
    uint64_t i8x8; //!< index for 4 8x8 blocks
    for (i8x8 = 0; i8x8 < n_mat/4*4; i8x8 += 4)
    {
        tbm_mult_t8x8_x4(in, in2, out);
        in += 4;
        in2 += 4;
        out += 4;
    }
    if (i8x8 == n_mat)
        return; // all blocks are processed
    uint64_t tmp[4];
    tbm_mult_t8x8_x4(in, in2, tmp);
    for (i8x8 = 0; i8x8 < (n_mat & 3); i8x8++)
        out[i8x8] = tmp[i8x8];
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
