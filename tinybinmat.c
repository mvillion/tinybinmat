#include "tinybinmat.h"

void tbm_print8(
    uint8_t *mat_list, uint64_t n_mat, uint8_t n_bit, char *str01)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        for (uint8_t i_row = 0; i_row < n_bit; i_row++)
        {
            uint8_t row = mat_list[i_row];
            for (uint8_t i_bit = 0; i_bit < n_bit; i_bit++)
            {
                printf("%c", str01[row & 1]);
                row >>= 1;
            }
            printf("\n");
        }
        mat_list += 8*sizeof(uint8_t);
        printf("\n");
    }
}

void tbm_print16(
    uint16_t *mat_list, uint64_t n_mat, uint8_t n_bit, char *str01)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        for (uint8_t i_row = 0; i_row < n_bit; i_row++)
        {
            uint16_t row = mat_list[i_row];
            for (uint8_t i_bit = 0; i_bit < n_bit; i_bit++)
            {
                printf("%c", str01[row & 1]);
                row >>= 1;
            }
            printf("\n");
        }
        mat_list += 8*sizeof(uint16_t);
        printf("\n");
    }
}

void tbm_sprint8(
    uint8_t *mat_list, uint64_t n_mat, uint8_t n_bit, char *str01,
    uint8_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        for (uint8_t i_row = 0; i_row < n_bit; i_row++)
        {
            uint8_t row = mat_list[i_row];
            for (uint8_t i_bit = 0; i_bit < n_bit; i_bit++)
            {
                *out++ = str01[row & 1];
                row >>= 1;
            }
        }
        mat_list += 8*sizeof(uint8_t);
    }
}

void tbm_sprint16(
    uint16_t *mat_list, uint64_t n_mat, uint8_t n_bit, char *str01,
    uint8_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        for (uint8_t i_row = 0; i_row < n_bit; i_row++)
        {
            uint16_t row = mat_list[i_row];
            for (uint8_t i_bit = 0; i_bit < n_bit; i_bit++)
            {
                *out++ = str01[row & 1];
                row >>= 1;
            }
        }
        mat_list += 8*sizeof(uint16_t);
    }
}

uint64_t inline tbm_transpose8x8_uint64(uint64_t in8x8)
{
    // input is 8x8 bit matrix with 8 rows: 0x0706050403020100
    // 1st bit in rows is LSB, thus reversed compared to matrix notation
    uint64_t ur_mask4x4 = 0x00000000f0f0f0f0; // up right 4x4 bits
    // uint64_t = dl_mask4x4 = 0x0f0f0f0f00000000; // down left 4x4 bits
    // dl_mask4x4 == ur_mask4x4 << (4*8-4)
    uint64_t xor = in8x8 ^ (in8x8 >> 28);
    xor &= ur_mask4x4;
    in8x8 ^= xor;
    xor <<= 28;
    in8x8 ^= xor;
    uint64_t ur_mask2x2 = 0x0000cccc0000cccc; // 4 up right 2x2 bits
    // uint64_t = dl_mask2x2 = 0x3333000033330000; // 4 down left 2x2 bits
    // dl_mask2x2 == ur_mask2x2 << (2*8-2)
    xor = in8x8 ^ (in8x8 >> 14);
    xor &= ur_mask2x2;
    in8x8 ^= xor;
    xor <<= 14;
    in8x8 ^= xor;
    uint64_t ur_mask1x1 = 0x00aa00aa00aa00aa; // 16 up right 1x1 bits
    // uint64_t = dl_mask1x1 = 0x5500550055005500; // 16 down left 1x1 bits
    // dl_mask1x1 == ur_mask1x1 << (8-1)
    xor = in8x8 ^ (in8x8 >> 7);
    xor &= ur_mask1x1;
    in8x8 ^= xor;
    xor <<= 7;
    in8x8 ^= xor;
    return in8x8;
}

void print_avx2_uint64(__m256i reg)
{
    uint64_t *ptr = (uint64_t *)&reg;
    for (uint8_t k = 0; k < 3; k++)
        printf("%016lx ", ptr[k]);
    printf("%016lx\n", ptr[3]);
}

#define USE_AVX2
#if defined(USE_AVX2)
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
#endif

void inline tbm_transpose16x16_uint64(
    uint64_t in03x16, uint64_t in47x16, uint64_t in8bx16, uint64_t incfx16,
    uint64_t *out03x16, uint64_t *out47x16, uint64_t *out8bx16, 
    uint64_t *outcfx16)
{
    // inputs are 2x16 bit matrix with 2 rows: 0x00000001_00000000
    // 1st bit in rows is LSB, thus reversed compared to matrix notation
    uint64_t ur_mask4x8 = 0xff00ff00ff00ff00; // up right 4x8 bits
    // uint64_t = dl_mask4x8 = 0x00ff00ff00ff00ff; // down left 4x8 bits
    // dl_mask4x8 == ur_mask4x8 >> 8
    uint64_t xor0 = in03x16 ^ (in8bx16 << 8);
    uint64_t xor1 = in47x16 ^ (incfx16 << 8);
    xor0 &= ur_mask4x8;
    xor1 &= ur_mask4x8;
    in03x16 ^= xor0;
    in47x16 ^= xor1;
    xor0 >>= 8;
    xor1 >>= 8;
    in8bx16 ^= xor0;
    incfx16 ^= xor1;
    uint64_t ur_mask4x4 = 0xf0f0f0f0f0f0f0f0; // 2 up right 4x4 bits
     // uint64_t = dl_mask4x4 = 0x0f0f0f0f0f0f0f0f; // 2 down left 4x4 bits
    // dl_mask4x4 == ur_mask4x4 >> 4
    xor0 = in03x16 ^ (in47x16 << 4);
    xor1 = in8bx16 ^ (incfx16 << 4);
    xor0 &= ur_mask4x4;
    xor1 &= ur_mask4x4;
    in03x16 ^= xor0;
    in8bx16 ^= xor1;
    xor0 >>= 4;
    xor1 >>= 4;
    in47x16 ^= xor0;
    incfx16 ^= xor1;
    uint64_t ur_mask2x2 = 0x00000000cccccccc; // 4 up right 2x2 bits
    // uint64_t = dl_mask2x2 = 0x3333333300000000; // 4 down left 2x2 bits
    // dl_mask2x2 == ur_mask2x2 << (16*2-2)
    xor0 = in03x16 ^ (in03x16 >> 30);
    xor0 &= ur_mask2x2; in03x16 ^= xor0; xor0 <<= 30; in03x16 ^= xor0;
    xor0 = in47x16 ^ (in47x16 >> 30);
    xor0 &= ur_mask2x2; in47x16 ^= xor0; xor0 <<= 30; in47x16 ^= xor0;
    xor0 = in8bx16 ^ (in8bx16 >> 30);
    xor0 &= ur_mask2x2; in8bx16 ^= xor0; xor0 <<= 30; in8bx16 ^= xor0;
    xor0 = incfx16 ^ (incfx16 >> 30);
    xor0 &= ur_mask2x2; incfx16 ^= xor0; xor0 <<= 30; incfx16 ^= xor0;
    uint64_t ur_mask1x1 = 0x0000aaaa0000aaaa; // 16 up right 1x1 bits
    // uint64_t = dl_mask1x1 = 0x5555000055550000; // 16 down left 1x1 bits
    // dl_mask1x1 == ur_mask1x1 << (16-1)
    xor0 = in03x16 ^ (in03x16 >> 15);
    xor0 &= ur_mask1x1; in03x16 ^= xor0; xor0 <<= 15; in03x16 ^= xor0;
    xor0 = in47x16 ^ (in47x16 >> 15);
    xor0 &= ur_mask1x1; in47x16 ^= xor0; xor0 <<= 15; in47x16 ^= xor0;
    xor0 = in8bx16 ^ (in8bx16 >> 15);
    xor0 &= ur_mask1x1; in8bx16 ^= xor0; xor0 <<= 15; in8bx16 ^= xor0;
    xor0 = incfx16 ^ (incfx16 >> 15);
    xor0 &= ur_mask1x1; incfx16 ^= xor0; xor0 <<= 15; incfx16 ^= xor0;
    *out03x16 = in03x16;
    *out47x16 = in47x16;
    *out8bx16 = in8bx16;
    *outcfx16 = incfx16;
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

#pragma GCC push_options //----------------------------------------------------
#pragma GCC optimize("no-tree-vectorize")
void tbm_transpose8x8(uint64_t *in8x8, uint64_t n_mat, uint64_t *out8x8)
{
    uint64_t i_avx2 = 0;
#if defined(USE_AVX2)
    i_avx2 = n_mat/4*4;
    for (uint64_t i_mat = 0; i_mat < i_avx2; i_mat += 4)
    {
        __m256i in8x8_4 = _mm256_loadu_si256((__m256i *)(in8x8+i_mat));
        __m256i out8x8_4 = tbm_transpose8x8_m256i(in8x8_4);
        _mm256_storeu_si256((__m256i *)(out8x8+i_mat), out8x8_4);
    }
#endif
    for (uint64_t i_mat = i_avx2; i_mat < n_mat; i_mat++)
        out8x8[i_mat] = tbm_transpose8x8_uint64(in8x8[i_mat]);
}

void tbm_transpose16x16(uint64_t *in2x16, uint64_t n_mat, uint64_t *out2x16)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
#if defined(USE_AVX2)
        __m256i in16x16 = _mm256_loadu_si256((__m256i *)in2x16);
        __m256i out16x16 = tbm_transpose16x16_m256i(in16x16);
        _mm256_storeu_si256((__m256i *)out2x16, out16x16);
#else
        tbm_transpose16x16_uint64(
            in2x16[0], in2x16[1], in2x16[2], in2x16[3],
            out2x16+0, out2x16+1, out2x16+2, out2x16+3);
#endif
        in2x16 += 4;
        out2x16 += 4;
    }
}
#pragma GCC pop_options //-----------------------------------------------------
