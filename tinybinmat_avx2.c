#include "tinybinmat.h"

//______________________________________________________________________________
__m256i inline tbm_transpose8x8_m256i(__m256i in8x8_4)
{
    __m256i ur_mask4x4 = _mm256_set1_epi64x(0x00000000f0f0f0f0);
    __m256i xor = _mm256_xor_si256(in8x8_4, _mm256_srli_epi64(in8x8_4, 28));
    xor = _mm256_and_si256(xor, ur_mask4x4);
    in8x8_4 = _mm256_xor_si256(in8x8_4, xor);
    xor = _mm256_slli_epi64(xor, 28);
    in8x8_4 = _mm256_xor_si256(in8x8_4, xor);
    __m256i ur_mask2x2 = _mm256_set1_epi64x(0x0000cccc0000cccc);
    xor = _mm256_xor_si256(in8x8_4, _mm256_srli_epi64(in8x8_4, 14));
    xor = _mm256_and_si256(xor, ur_mask2x2);
    in8x8_4 = _mm256_xor_si256(in8x8_4, xor);
    xor = _mm256_slli_epi64(xor, 14);
    in8x8_4 = _mm256_xor_si256(in8x8_4, xor);
    __m256i ur_mask1x1 = _mm256_set1_epi64x(0x00aa00aa00aa00aa);
    xor = _mm256_xor_si256(in8x8_4, _mm256_srli_epi64(in8x8_4, 7));
    xor = _mm256_and_si256(xor, ur_mask1x1);
    in8x8_4 = _mm256_xor_si256(in8x8_4, xor);
    xor = _mm256_slli_epi64(xor, 7);
    in8x8_4 = _mm256_xor_si256(in8x8_4, xor);
    return in8x8_4;
}

__m256i inline tbm_transpose16x16_m256i(__m256i in16x16)
{
    __m256i ur_mask8x8 = _mm256_set_epi64x(
        0, 0, 0xff00ff00ff00ff00, 0xff00ff00ff00ff00);
    // we want to shift right in16x16 by 120
    // we first shift right by 128, then shift left by 8
    __m256i roll128 = _mm256_permute2x128_si256(in16x16, in16x16, 0x01);
    __m256i xor = _mm256_xor_si256(in16x16, _mm256_bslli_epi128(roll128, 1));
    xor = _mm256_and_si256(xor, ur_mask8x8);
    in16x16 = _mm256_xor_si256(in16x16, xor);
    // we want to shift left xor by 120, xor is all in lower 128 bits
    // we first shift left by 128, then shift right by 8
    roll128 = _mm256_permute2x128_si256(xor, xor, 0x01);
    xor = _mm256_bsrli_epi128(roll128, 1);
    in16x16 = _mm256_xor_si256(in16x16, xor);
    __m256i ur_mask4x4 = _mm256_set_epi64x(
        0, 0xf0f0f0f0f0f0f0f0, 0, 0xf0f0f0f0f0f0f0f0);
    // we want to shift right in16x16 by 60 inside 128-bit lanes
    // we first shift right by 8*8, then shift left by 4
    __m256i shift64 = _mm256_bsrli_epi128(in16x16, 8);
    xor = _mm256_xor_si256(in16x16, _mm256_slli_epi64(shift64, 4));
    xor = _mm256_and_si256(xor, ur_mask4x4);
    in16x16 = _mm256_xor_si256(in16x16, xor);
    // we want to shift left by 60 inside 128-bit lanes
    // we first shift left by 8*8, then shift right by 4
    xor = _mm256_bslli_epi128(xor, 8);
    xor = _mm256_srli_epi64(xor, 4);
    in16x16 = _mm256_xor_si256(in16x16, xor);
    __m256i ur_mask2x2 = _mm256_set1_epi64x(0x00000000cccccccc);
    xor = _mm256_xor_si256(in16x16, _mm256_srli_epi64(in16x16, 30));
    xor = _mm256_and_si256(xor, ur_mask2x2);
    in16x16 = _mm256_xor_si256(in16x16, xor);
    xor = _mm256_slli_epi64(xor, 30);
    in16x16 = _mm256_xor_si256(in16x16, xor);
    __m256i ur_mask1x1 = _mm256_set1_epi64x(0x0000aaaa0000aaaa);
    xor = _mm256_xor_si256(in16x16, _mm256_srli_epi64(in16x16, 15));
    xor = _mm256_and_si256(xor, ur_mask1x1);
    in16x16 = _mm256_xor_si256(in16x16, xor);
    xor = _mm256_slli_epi64(xor, 15);
    in16x16 = _mm256_xor_si256(in16x16, xor);
    return in16x16;
}

void inline tbm_transpose32x32_m256i(__m256i in_read[4], __m256i in[4])
{
    for (uint8_t i_8row = 0; i_8row < 4; i_8row++)
        in[i_8row] = in_read[i_8row];

    __m256i xor;
#if 0
    __m256i ur_mask8x16 = _mm256_set1_epi64x(0xffff0000ffff0000);
    for (uint8_t i_8row = 0; i_8row < 2; i_8row++)
    {
        xor = _mm256_xor_si256(
            in[i_8row], _mm256_bslli_epi128(in[i_8row+2], 2));
        xor = _mm256_and_si256(xor, ur_mask8x16);
        in[i_8row] = _mm256_xor_si256(in[i_8row], xor);
        xor = _mm256_bsrli_epi128(xor, 2);
        in[i_8row+2] = _mm256_xor_si256(in[i_8row+2], xor);
    }
#else
    for (uint8_t i_8row = 0; i_8row < 2; i_8row++)
    {
        __m256i in0 = in[i_8row];
        __m256i in1 = in[i_8row+2];
        __m256i in0_sr16 = _mm256_srli_epi32(in0, 16);
        __m256i in1_sl16 = _mm256_slli_epi32(in1, 16);
        in[i_8row] = _mm256_blend_epi16(in0, in1_sl16, 0xaa);
        in[i_8row+2] = _mm256_blend_epi16(in1, in0_sr16, 0x55);
    }
#endif
    __m256i ur_mask8x8 = _mm256_set1_epi64x(0xff00ff00ff00ff00);
    for (uint8_t i_block = 0; i_block < 2; i_block++)
    {
        uint8_t i_8row = i_block*2;
        xor = _mm256_xor_si256(
            in[i_8row], _mm256_bslli_epi128(in[i_8row+1], 1));
        xor = _mm256_and_si256(xor, ur_mask8x8);
        in[i_8row] = _mm256_xor_si256(in[i_8row], xor);
        xor = _mm256_bsrli_epi128(xor, 1);
        in[i_8row+1] = _mm256_xor_si256(in[i_8row+1], xor);
    }
    __m256i ur_mask4x4 = _mm256_set_epi64x(
        0, 0, 0xf0f0f0f0f0f0f0f0, 0xf0f0f0f0f0f0f0f0);
    for (uint8_t i_8row = 0; i_8row < 4; i_8row++)
    {
        // we want to shift right ur_mask4x4 by 124
        // we first shift right by 128, then shift left by 4
        __m256i roll128 = _mm256_permute2x128_si256(
            in[i_8row], in[i_8row], 0x01);
        xor = _mm256_xor_si256(in[i_8row], _mm256_slli_epi64(roll128, 4));
        xor = _mm256_and_si256(xor, ur_mask4x4);
        in[i_8row] = _mm256_xor_si256(in[i_8row], xor);
        // we want to shift left xor by 124, xor is all in lower 128 bits
        // we first shift left by 128, then shift right by 4
        roll128 = _mm256_permute2x128_si256(xor, xor, 0x01);
        xor = _mm256_srli_epi64(roll128, 4);
        in[i_8row] = _mm256_xor_si256(in[i_8row], xor);
    }
    __m256i ur_mask2x2 = _mm256_set_epi64x(
        0, 0xcccccccccccccccc, 0, 0xcccccccccccccccc);
    for (uint8_t i_8row = 0; i_8row < 4; i_8row++)
    {
        // we want to shift right in16x16 by 62 inside 128-bit lanes
        // we first shift right by 8*8, then shift left by 2
        __m256i shift64 = _mm256_bsrli_epi128(in[i_8row], 8);
        xor = _mm256_xor_si256(in[i_8row], _mm256_slli_epi64(shift64, 2));
        xor = _mm256_and_si256(xor, ur_mask2x2);
        in[i_8row] = _mm256_xor_si256(in[i_8row], xor);
        // we want to shift left xor by 62, xor is all in lower 128 bits
        // we first shift left by 64, then shift right by 2
        xor = _mm256_bslli_epi128(xor, 8);
        xor = _mm256_srli_epi64(xor, 2);
        in[i_8row] = _mm256_xor_si256(in[i_8row], xor);
    }
    __m256i ur_mask1x1 = _mm256_set1_epi64x(0x00000000aaaaaaaa);
    for (uint8_t i_8row = 0; i_8row < 4; i_8row++)
    {
        xor = _mm256_xor_si256(in[i_8row], _mm256_srli_epi64(in[i_8row], 31));
        xor = _mm256_and_si256(xor, ur_mask1x1);
        in[i_8row] = _mm256_xor_si256(in[i_8row], xor);
        xor = _mm256_slli_epi64(xor, 31);
        in[i_8row] = _mm256_xor_si256(in[i_8row], xor);
    }
}

#pragma GCC push_options //----------------------------------------------------
#pragma GCC optimize("no-tree-vectorize")
void tbm_transpose8x8_avx2(uint8_t *in, uint64_t n_mat, uint8_t *out)
{
    uint64_t i_avx2 = n_mat/4*4;
    for (uint64_t i_mat = 0; i_mat < i_avx2; i_mat += 4)
    {
        __m256i in8x8_4 = _mm256_loadu_si256((__m256i *)in);
        __m256i out8x8_4 = tbm_transpose8x8_m256i(in8x8_4);
        _mm256_storeu_si256((__m256i *)out, out8x8_4);
        in += 8*4;
        out += 8*4;
    }
    for (uint64_t i_mat = i_avx2; i_mat < n_mat; i_mat++)
    {
        *((uint64_t *)out) = tbm_transpose8x8_uint64(*((uint64_t *)in));
        in += 8;
        out += 8;
    }
}

void tbm_transpose16x16_avx2(uint16_t *in, uint64_t n_mat, uint16_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        __m256i in16x16 = _mm256_loadu_si256((__m256i *)in);
        __m256i out16x16 = tbm_transpose16x16_m256i(in16x16);
        _mm256_storeu_si256((__m256i *)out, out16x16);
        in += 16;
        out += 16;
    }
}

void tbm_transpose32x32_avx2(uint32_t *in, uint64_t n_mat, uint32_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        __m256i in8x32[4];
        for (uint8_t i_8row = 0; i_8row < 4; i_8row++)
            in8x32[i_8row] = _mm256_loadu_si256(((__m256i *)in)+i_8row);
        tbm_transpose32x32_m256i(in8x32, in8x32);
        for (uint8_t i_8row = 0; i_8row < 4; i_8row++)
            _mm256_storeu_si256(((__m256i *)out)+i_8row, in8x32[i_8row]);
        in += 32;
        out += 32;
    }
}
#pragma GCC pop_options //-----------------------------------------------------

//______________________________________________________________________________
// multiply 4 groups of two 8x8 bit matrices
__m256i inline tbm_mult8x8_m256i(__m256i a, uint8_t b[32])
{
    __m128i repeat8x2 = _mm_set_epi8(
        8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0);
    __m256i repeat8x4 = _mm256_set_m128i(repeat8x2, repeat8x2);
    
    __m256i out = _mm256_setzero_si256();
    __m256i test_bit = _mm256_set1_epi8(1);
    for (uint8_t i_bit = 0; i_bit < 8; i_bit++)
    {
        // create bit mask from the least significant bits in a
        __m256i bit_a = _mm256_and_si256(a, test_bit);
        bit_a = _mm256_cmpeq_epi8(bit_a, test_bit);
        test_bit = _mm256_add_epi8(test_bit, test_bit);
        // load 32 octets from b starting from i_bit
        __m256i b_i_bit32 = _mm256_loadu_si256((__m256i *)(b+i_bit));
        b_i_bit32 = _mm256_shuffle_epi8(b_i_bit32, repeat8x4);
        __m256i prod = _mm256_and_si256(bit_a, b_i_bit32);
        out = _mm256_xor_si256(out, prod);
    }
    return out;
}

__m256i inline tbm_mult16x16_m256i(__m256i a, uint16_t b[16])
{
    __m256i out = _mm256_setzero_si256();
    // __m256i test_bit = _mm256_set1_epi16(1);
    for (uint8_t i_bit = 0; i_bit < 16; i_bit++)
    {
#if 1
        // create bit mask from the most significant bits in a
        __m256i bit_a = _mm256_srai_epi16(a, 16);
        a = _mm256_slli_epi16(a, 1);
        __m256i prod = _mm256_and_si256(bit_a, _mm256_set1_epi16(b[15-i_bit]));
#else
        // create bit mask from the least significant bits in a
        __m256i bit_a = _mm256_and_si256(a, test_bit);
        bit_a = _mm256_cmpeq_epi16(bit_a, test_bit);
        test_bit = _mm256_slli_epi16(test_bit, 1);
        __m256i prod = _mm256_and_si256(bit_a, _mm256_set1_epi16(b[i_bit]));
#endif
        out = _mm256_xor_si256(out, prod);
    }
    return out;
}

void inline tbm_mult32x32_m256i(__m256i a[4], uint32_t b[32], __m256i out[4])
{
    for (uint8_t i_8row = 0; i_8row < 4; i_8row++)
        out[i_8row] = _mm256_setzero_si256();

#if 0
    __m256i a_bck[4];
    for (uint8_t i_8row = 0; i_8row < 4; i_8row++)
        a_bck[i_8row] = a[i_8row];

    for (uint8_t i_bit = 0; i_bit < 32; i_bit++)
        for (uint8_t i_8row = 0; i_8row < 4; i_8row++)
        {
            // create bit mask from the most significant bits in a
            __m256i bit_a = _mm256_srai_epi32(a_bck[i_8row], 32);
            a_bck[i_8row] = _mm256_slli_epi32(a_bck[i_8row], 1);
            __m256i prod = _mm256_and_si256(
                bit_a, _mm256_set1_epi32(b[31-i_bit]));
            out[i_8row] = _mm256_xor_si256(out[i_8row], prod);
        }
#else
    __m256i test_bit0 = _mm256_set1_epi32(1);
    __m256i test_bit16 = _mm256_set1_epi32(0x10000);
    for (uint8_t i_bit = 0; i_bit < 16; i_bit++)
    {
        for (uint8_t i_8row = 0; i_8row < 4; i_8row++)
        {
            // create bit mask from the least significant bits in a
            __m256i bit_a = _mm256_and_si256(a[i_8row], test_bit0);
            bit_a = _mm256_cmpeq_epi32(bit_a, test_bit0);
            __m256i prod = _mm256_and_si256(bit_a, _mm256_set1_epi32(b[i_bit]));
            out[i_8row] = _mm256_xor_si256(out[i_8row], prod);

            bit_a = _mm256_and_si256(a[i_8row], test_bit16);
            bit_a = _mm256_cmpeq_epi32(bit_a, test_bit16);
            prod = _mm256_and_si256(bit_a, _mm256_set1_epi32(b[i_bit+16]));
            out[i_8row] = _mm256_xor_si256(out[i_8row], prod);
        }
        test_bit0 = _mm256_slli_epi32(test_bit0, 1);
        test_bit16 = _mm256_slli_epi32(test_bit16, 1);
    }
#endif
}

#pragma GCC push_options //----------------------------------------------------
#pragma GCC optimize("no-tree-vectorize")
void tbm_mult8x8_avx2(uint8_t *in, uint8_t *in2, uint64_t n_mat, uint8_t *out)
{
    uint64_t i_avx2 = 0;
    i_avx2 = n_mat/4*4;
    for (uint64_t i_mat = 0; i_mat < i_avx2; i_mat += 4)
    {
        __m256i in8x8_4 = _mm256_loadu_si256((__m256i *)in);
        __m256i out8x8_4 = tbm_mult8x8_m256i(in8x8_4, in2);
        _mm256_storeu_si256((__m256i *)out, out8x8_4);
        in += 8*4;
        in2 += 8*4;
        out += 8*4;
    }
    for (uint64_t i_mat = i_avx2; i_mat < n_mat; i_mat++)
    {
        *((uint64_t *)out) = tbm_mult8x8_uint64(*((uint64_t *)in), in2);
        in += 8;
        in2 += 8;
        out += 8;
    }
}

void tbm_mult16x16_avx2(
    uint16_t *in, uint16_t *in2, uint64_t n_mat, uint16_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        __m256i in16x16 = _mm256_loadu_si256((__m256i *)in);
        __m256i out16x16 = tbm_mult16x16_m256i(in16x16, in2);
        _mm256_storeu_si256((__m256i *)out, out16x16);
        in += 16;
        in2 += 16;
        out += 16;
    }
}

void tbm_mult32x32_avx2(
    uint32_t *in, uint32_t *in2, uint64_t n_mat, uint32_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        __m256i in8x32[4];
        __m256i out8x32[4];
        for (uint8_t i_8row = 0; i_8row < 4; i_8row++)
            in8x32[i_8row] = _mm256_loadu_si256(((__m256i *)in)+i_8row);
        tbm_mult32x32_m256i(in8x32, in2, out8x32);
        for (uint8_t i_8row = 0; i_8row < 4; i_8row++)
            _mm256_storeu_si256(((__m256i *)out)+i_8row, out8x32[i_8row]);
        in += 32;
        in2 += 32;
        out += 32;
    }
}
#pragma GCC pop_options //-----------------------------------------------------

//______________________________________________________________________________
// multiply two 8x8 bit matrices with the second matrix transposed
uint64_t inline tbm_mult_t8x8_single_m256i(uint64_t a8x8, uint64_t tb8x8)
{
    // note: this code output is transposed, thus input were swapped...
    __m256i a8x8_4 = _mm256_set1_epi64x(tb8x8);
    __m256i row_b_4 = _mm256_cvtepu8_epi64(_mm_set_epi64x(0, a8x8 >> 32));
    __m128i mask = _mm_set_epi8(8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0);
    __m256i repeat_mask = _mm256_set_m128i(mask, mask);

    __m256i repeat_4 = _mm256_shuffle_epi8(row_b_4, repeat_mask);
    __m256i prod_4 = _mm256_and_si256(a8x8_4, repeat_4);
    prod_4 = _mm256_xor_si256(prod_4, _mm256_slli_epi16(prod_4, 4));
    prod_4 = _mm256_xor_si256(prod_4, _mm256_slli_epi16(prod_4, 2));
    prod_4 = _mm256_xor_si256(prod_4, _mm256_slli_epi16(prod_4, 1));
    uint64_t out = (uint32_t)_mm256_movemask_epi8(prod_4);
    
    row_b_4 = _mm256_cvtepu8_epi64(_mm_set_epi64x(0, a8x8));
    repeat_4 = _mm256_shuffle_epi8(row_b_4, repeat_mask);
    prod_4 = _mm256_and_si256(a8x8_4, repeat_4);
    prod_4 = _mm256_xor_si256(prod_4, _mm256_slli_epi16(prod_4, 4));
    prod_4 = _mm256_xor_si256(prod_4, _mm256_slli_epi16(prod_4, 2));
    prod_4 = _mm256_xor_si256(prod_4, _mm256_slli_epi16(prod_4, 1));
    out <<= 32;
    out |= (uint32_t)_mm256_movemask_epi8(prod_4);
    return out;
}

__m256i inline tbm_mult_t16x16_m256i(__m256i tb16x16, uint16_t a1x16[16])
{
    __m256i prod[8];
    for (uint8_t i_row = 0; i_row < 8; i_row++)
    {
        __m256i prodl = _mm256_and_si256(
            tb16x16, _mm256_set1_epi16(a1x16[i_row]));
        // prodl ^= prodl << 8; prodl >>= 8;
        // xored values are in lower octets of epi16, upper octets are 0
        prodl = _mm256_xor_si256(prodl, _mm256_slli_epi16(prodl, 8));
        prodl = _mm256_srli_epi16(prodl, 8);
        __m256i produ = _mm256_and_si256(
            tb16x16, _mm256_set1_epi16(a1x16[i_row+8]));
        // produ ^= produ >> 8; produ <<= 8;
        // xored values are in upper octets of epi16, lower octets are 0
        produ = _mm256_xor_si256(produ, _mm256_srli_epi16(produ, 8));
        produ = _mm256_slli_epi16(produ, 8);
        prod[i_row] = _mm256_or_si256(prodl, produ);
    }
    for (uint8_t i_row = 0; i_row < 4; i_row++)
    {
        __m256i prodl = prod[i_row];
        __m256i mask = _mm256_set1_epi8(0x0f);
        // prodl ^= prodl >> 4;
        // xored values are in lower 4-bit of octets, upper 4-bit are 0
        prodl = _mm256_xor_si256(prodl, _mm256_srli_epi16(prodl, 4));
        prodl = _mm256_and_si256(mask, prodl);
        __m256i produ = prod[i_row+4];
        // produ ^= produ << 4;
        // xored values are in upper 4-bit of octets, lower 4-bit are 0
        produ = _mm256_xor_si256(produ, _mm256_slli_epi16(produ, 4));
        produ = _mm256_andnot_si256(mask, produ);
        prod[i_row] = _mm256_or_si256(prodl, produ);
    }
    for (uint8_t i_row = 0; i_row < 2; i_row++)
    {
        __m256i prodl = prod[i_row];
       __m256i mask = _mm256_set1_epi8(0x33);
        // prodl ^= prodl >> 2;
        // xored values are in lower 2-bit, upper 2-bit are 0
        prodl = _mm256_xor_si256(prodl, _mm256_srli_epi16(prodl, 2));
        prodl = _mm256_and_si256(mask, prodl);
        __m256i produ = prod[i_row+2];
        // produ ^= produ << 2;
        // xored values are in upper 2-bit, lower 2-bit are 0
        produ = _mm256_xor_si256(produ, _mm256_slli_epi16(produ, 2));
        produ = _mm256_andnot_si256(mask, produ);
        prod[i_row] = _mm256_or_si256(prodl, produ);
    }
    for (uint8_t i_row = 0; i_row < 1; i_row++)
    {
        __m256i prodl = prod[i_row];
        __m256i mask = _mm256_set1_epi8(0x55);
        // prodl ^= prodl >> 1;
        // xored values are in lower 1-bit, upper 1-bit is 0
        prodl = _mm256_xor_si256(prodl, _mm256_srli_epi16(prodl, 1));
        prodl = _mm256_and_si256(mask, prodl);
        __m256i produ = prod[i_row+1];
        // produ ^= produ << 1;
        // xored values are in upper 1-bit, lower 1-bit is 0
        produ = _mm256_xor_si256(produ, _mm256_slli_epi16(produ, 1));
        produ = _mm256_andnot_si256(mask, produ);
        prod[i_row] = _mm256_or_si256(prodl, produ);
    }
    return prod[0];
}

void inline tbm_mult_t32x32_m256i(__m256i tb8x32[4], uint32_t a1x32[32])
{
    for (uint8_t i_8col = 0; i_8col < 4; i_8col++)
    {
        __m256i prod[16];
        for (uint8_t i_row = 0; i_row < 16; i_row++)
        {
            __m256i prodl = _mm256_and_si256(
                tb8x32[i_8col], _mm256_set1_epi32(a1x32[i_row]));
            // prodl ^= prodl >> 16;
            // xored values in lower 16-bit of epi32, upper 16-bit are invalid
            prodl = _mm256_xor_si256(prodl, _mm256_srli_epi32(prodl, 16));
            __m256i produ = _mm256_and_si256(
                tb8x32[i_8col], _mm256_set1_epi32(a1x32[i_row+16]));
            // produ ^= produ << 16;
            // xored values in upper 16-bit of epi32, lower 16-bit are invalid
            produ = _mm256_xor_si256(produ, _mm256_slli_epi32(produ, 16));
            prod[i_row] = _mm256_blend_epi16(prodl, produ, 0xaa);
        }
        for (uint8_t i_row = 0; i_row < 8; i_row++)
        {
            __m256i prodl = prod[i_row];
            prodl = _mm256_xor_si256(prodl, _mm256_slli_epi16(prodl, 8));
            prodl = _mm256_srli_epi16(prodl, 8);
            __m256i produ = prod[i_row+8];
            produ = _mm256_xor_si256(produ, _mm256_srli_epi16(produ, 8));
            produ = _mm256_slli_epi16(produ, 8);
            prod[i_row] = _mm256_or_si256(prodl, produ);
        }
        for (uint8_t i_row = 0; i_row < 4; i_row++)
        {
            __m256i prodl = prod[i_row];
            __m256i mask = _mm256_set1_epi8(0x0f);
            // prodl ^= prodl >> 4;
            // xored values are in lower 4-bit of octets, upper 4-bit are 0
            prodl = _mm256_xor_si256(prodl, _mm256_srli_epi16(prodl, 4));
            prodl = _mm256_and_si256(mask, prodl);
            __m256i produ = prod[i_row+4];
            // produ ^= produ << 4;
            // xored values are in upper 4-bit of octets, lower 4-bit are 0
            produ = _mm256_xor_si256(produ, _mm256_slli_epi16(produ, 4));
            produ = _mm256_andnot_si256(mask, produ);
            prod[i_row] = _mm256_or_si256(prodl, produ);
        }
        for (uint8_t i_row = 0; i_row < 2; i_row++)
        {
            __m256i prodl = prod[i_row];
        __m256i mask = _mm256_set1_epi8(0x33);
            // prodl ^= prodl >> 2;
            // xored values are in lower 2-bit, upper 2-bit are 0
            prodl = _mm256_xor_si256(prodl, _mm256_srli_epi16(prodl, 2));
            prodl = _mm256_and_si256(mask, prodl);
            __m256i produ = prod[i_row+2];
            // produ ^= produ << 2;
            // xored values are in upper 2-bit, lower 2-bit are 0
            produ = _mm256_xor_si256(produ, _mm256_slli_epi16(produ, 2));
            produ = _mm256_andnot_si256(mask, produ);
            prod[i_row] = _mm256_or_si256(prodl, produ);
        }
        for (uint8_t i_row = 0; i_row < 1; i_row++)
        {
            __m256i prodl = prod[i_row];
            __m256i mask = _mm256_set1_epi8(0x55);
            // prodl ^= prodl >> 1;
            // xored values are in lower 1-bit, upper 1-bit is 0
            prodl = _mm256_xor_si256(prodl, _mm256_srli_epi16(prodl, 1));
            prodl = _mm256_and_si256(mask, prodl);
            __m256i produ = prod[i_row+1];
            // produ ^= produ << 1;
            // xored values are in upper 1-bit, lower 1-bit is 0
            produ = _mm256_xor_si256(produ, _mm256_slli_epi16(produ, 1));
            produ = _mm256_andnot_si256(mask, produ);
            prod[i_row] = _mm256_or_si256(prodl, produ);
        }
        tb8x32[i_8col] = prod[0];
    }
}

#pragma GCC push_options //----------------------------------------------------
#pragma GCC optimize("no-tree-vectorize")
void tbm_mult_t8x8_avx2(
    uint8_t *in, uint8_t *in2t, uint64_t n_mat, uint8_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        *((uint64_t *)out) = tbm_mult_t8x8_single_m256i(
            *((uint64_t *)in), *((uint64_t *)in2t));
        in += 8;
        in2t += 8;
        out += 8;
    }
}

void tbm_mult_t16x16_avx2(
    uint16_t *in, uint16_t *in2t, uint64_t n_mat, uint16_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        __m256i in16x16 = _mm256_loadu_si256((__m256i *)in);
        __m256i out16x16 = tbm_mult_t16x16_m256i(in16x16, in2t);
        _mm256_storeu_si256((__m256i *)out, out16x16);
        in += 16;
        in2t += 16;
        out += 16;
    }
}

void tbm_mult_t32x32_avx2(
    uint32_t *in, uint32_t *in2t, uint64_t n_mat, uint32_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        __m256i in8x32[4];
        for (uint8_t i_8row = 0; i_8row < 4; i_8row++)
            in8x32[i_8row] = _mm256_loadu_si256(((__m256i *)in)+i_8row);
        tbm_mult_t32x32_m256i(in8x32, in2t);
        for (uint8_t i_8row = 0; i_8row < 4; i_8row++)
            _mm256_storeu_si256(((__m256i *)out)+i_8row, in8x32[i_8row]);
        in += 32;
        in2t += 32;
        out += 32;
}
}
#pragma GCC pop_options //-----------------------------------------------------
