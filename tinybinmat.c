#include "tinybinmat.h"

void print_avx2_uint64(__m256i reg)
{
    uint64_t *ptr = (uint64_t *)&reg;
    for (uint8_t k = 0; k < 3; k++)
        printf("%016lx ", ptr[k]);
    printf("%016lx\n", ptr[3]);
}

__m256i _mm256_movm_epi8_avx2(const uint32_t mask) 
{
    __m256i vmask = _mm256_set1_epi32(mask);
    const __m256i shuffle = _mm256_set_epi64x(
        0x0303030303030303, 0x0202020202020202,
        0x0101010101010101, 0x0000000000000000);
    vmask = _mm256_shuffle_epi8(vmask, shuffle);
    // "%016x" % (0x7fbfdfeff7fbfdfe ^ ((2 << 64)-1)) -> '18040201008040201'
    const __m256i bit_mask = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe);
    vmask = _mm256_or_si256(vmask, bit_mask);
    return _mm256_cmpeq_epi8(vmask, _mm256_set1_epi64x(-1));
}

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

void tbm_sprint8_avx2(
    uint8_t *mat_list, uint64_t n_mat, char *str01, uint8_t *out)
{
    uint64_t *mat_list64 = (uint64_t *)mat_list;
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        __m256i mask32 = _mm256_movm_epi8_avx2(mat_list64[i_mat] & 0xffffffff);
        _mm256_storeu_si256((__m256i *)out, mask32);
        out += 8*4;
        mask32 = _mm256_movm_epi8_avx2(mat_list64[i_mat] >> 32);
        _mm256_storeu_si256((__m256i *)out, mask32);
        out += 8*4;
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

void tbm_sprint32(
    uint32_t *mat_list, uint64_t n_mat, uint8_t n_bit, char *str01,
    uint8_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        for (uint8_t i_row = 0; i_row < n_bit; i_row++)
        {
            uint32_t row = mat_list[i_row];
            for (uint8_t i_bit = 0; i_bit < n_bit; i_bit++)
            {
                *out++ = str01[row & 1];
                row >>= 1;
            }
        }
        mat_list += 8*sizeof(uint32_t);
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

void inline tbm_transpose16x16_uint64(
    uint64_t in03x16, uint64_t in47x16, uint64_t in8bx16, uint64_t incfx16,
    uint64_t *out03x16, uint64_t *out47x16, uint64_t *out8bx16, 
    uint64_t *outcfx16)
{
    // inputs are 4x16 bit matrix with 4 rows: 0x00030002_00010000
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

void inline tbm_transpose32x32_uint64(uint64_t in_read[16], uint64_t in[16])
{
    for (uint8_t i_row = 0; i_row < 16; i_row++)
        in[i_row] = in_read[i_row];
    // inputs are 2x32 bit matrix with 2 rows: 0x00000001_00000000
    // 1st bit in rows is LSB, thus reversed compared to matrix notation
    uint64_t ur_mask2x16 = 0xffff0000ffff0000; // up right 2x16 bits
    // uint64_t = dl_mask2x16 = 0x0000ffff0000ffff; // down left 2x16 bits
    // dl_mask2x16 == ur_mask2x16 >> 16
    uint64_t xor;
    for (uint8_t i_row = 0; i_row < 8; i_row++)
    {
        xor = in[i_row] ^ (in[i_row+8] << 16);
        xor &= ur_mask2x16;
        in[i_row] ^= xor;
        xor >>= 16;
        in[i_row+8] ^= xor;
    }
    uint64_t ur_mask2x8 = 0xff00ff00ff00ff00; // 2 up right 2x8 bits
    // uint64_t = dl_mask2x8 = 0x00ff00ff00ff00ff; // 2 down left 2x16 bits
    // dl_mask2x8 == ur_mask2x8 >> 8
    for (uint8_t i_block = 0; i_block < 2; i_block++)
        for (uint8_t i_row = 0; i_row < 4; i_row++)
        {
            xor = in[i_block*8+i_row] ^ (in[i_block*8+i_row+4] << 8);
            xor &= ur_mask2x8;
            in[i_block*8+i_row] ^= xor;
            xor >>= 8;
            in[i_block*8+i_row+4] ^= xor;
        }
    uint64_t ur_mask2x4 = 0xf0f0f0f0f0f0f0f0; // 4 up right 2x4 bits
    // uint64_t = dl_mask2x4 = 0x0f0f0f0f0f0f0f0f; // 4 down left 2x4 bits
    // dl_mask2x4 == ur_mask2x4 >> 4
    for (uint8_t i_block = 0; i_block < 4; i_block++)
        for (uint8_t i_row = 0; i_row < 2; i_row++)
        {
            xor = in[i_block*4+i_row] ^ (in[i_block*4+i_row+2] << 4);
            xor &= ur_mask2x4;
            in[i_block*4+i_row] ^= xor;
            xor >>= 4;
            in[i_block*4+i_row+2] ^= xor;
        }
    uint64_t ur_mask2x2 = 0xcccccccccccccccc; // 8 up right 2x2 bits
    // uint64_t = dl_mask2x2 = 0x3333333333333333; // 4 down left 2x2 bits
    // dl_mask2x2 == ur_mask2x2 >> 2
    for (uint8_t i_block = 0; i_block < 8; i_block++)
        for (uint8_t i_row = 0; i_row < 1; i_row++)
        {
            xor = in[i_block*2+i_row] ^ (in[i_block*2+i_row+1] << 2);
            xor &= ur_mask2x2;
            in[i_block*2+i_row] ^= xor;
            xor >>= 2;
            in[i_block*2+i_row+1] ^= xor;
        }
    uint64_t ur_mask1x1 = 0x00000000aaaaaaaa; // 16 up right 1x1 bits
    // uint64_t = dl_mask1x1 = 0x5555555500000000; // 16 down left 1x1 bits
    // dl_mask1x1 == ur_mask1x1 << (32-1)
    for (uint8_t i_row = 0; i_row < 16; i_row++)
    {
        xor = in[i_row] ^ (in[i_row] >> 31);
        xor &= ur_mask1x1;
        in[i_row] ^= xor; xor <<= 31; in[i_row] ^= xor;
    }
    return;
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

#define USE_AVX2
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

void tbm_transpose16x16(uint64_t *in4x16, uint64_t n_mat, uint64_t *out4x16)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
#if defined(USE_AVX2)
        __m256i in16x16 = _mm256_loadu_si256((__m256i *)in4x16);
        __m256i out16x16 = tbm_transpose16x16_m256i(in16x16);
        _mm256_storeu_si256((__m256i *)out4x16, out16x16);
#else
        tbm_transpose16x16_uint64(
            in4x16[0], in4x16[1], in4x16[2], in4x16[3],
            out4x16+0, out4x16+1, out4x16+2, out4x16+3);
#endif
        in4x16 += 4;
        out4x16 += 4;
    }
}

void tbm_transpose32x32(uint64_t *in2x32, uint64_t n_mat, uint64_t *out2x32)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
#if defined(USE_AVX2)
        __m256i in8x32[4];
        for (uint8_t i_8row = 0; i_8row < 4; i_8row++)
            in8x32[i_8row] = _mm256_loadu_si256(((__m256i *)in2x32)+i_8row);
        tbm_transpose32x32_m256i(in8x32, in8x32);
        for (uint8_t i_8row = 0; i_8row < 4; i_8row++)
            _mm256_storeu_si256(((__m256i *)out2x32)+i_8row, in8x32[i_8row]);
#else
        tbm_transpose32x32_uint64(in2x32, out2x32);
#endif
        in2x32 += 16;
        out2x32 += 16;
    }
}
#pragma GCC pop_options //-----------------------------------------------------

// mult two 8x8 bit matrices with the second matrix transposed
uint64_t inline tbm_mult_t8x8_uint64(uint64_t a8x8, uint64_t tb8x8)
{
    uint64_t out = 0;
    for (uint8_t i_bit = 0; i_bit < 8; i_bit++)
    {
        uint8_t row_b = tb8x8 & 0xff;
        tb8x8 >>= 8;
        uint64_t repeat = 0x0101010101010101*row_b;
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

uint64_t inline tbm_mult_t8x8_m256i(uint64_t a8x8, uint64_t tb8x8)
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

// mult two 16x16 bit matrices with the second matrix transposed
// note: this code output is transposed, thus input were swapped...
void inline tbm_mult_t16x16_uint64(
    uint64_t tb4x16[4], uint64_t a4x16[4], uint64_t out4x16[4])
{
    for (uint8_t i_4col = 0; i_4col < 4; i_4col++)
    { 
        uint64_t out = 0;
        uint64_t tb_4col = tb4x16[i_4col];
        uint64_t a_4row[4];
        for (uint8_t i_4row = 0; i_4row < 4; i_4row++)
            a_4row[i_4row] = a4x16[i_4row];
        uint64_t prod[4];
        for (uint8_t i_bit = 0; i_bit < 4; i_bit++)
        {
            uint64_t row_a = a_4row[0] & 0xffff;
            a_4row[0] >>= 16;
            uint64_t repeat = 0x0001000100010001*row_a;
            prod[0] = tb_4col & repeat;
            prod[0] ^= prod[0] >> 8;
            prod[0] &= 0x00ff00ff00ff00ff;
            row_a = a_4row[2] & 0xffff;
            a_4row[2] >>= 16;
            repeat = 0x0001000100010001*row_a;
            prod[2] = tb_4col & repeat;
            prod[2] ^= prod[2] >> 8;
            prod[2] &= 0x00ff00ff00ff00ff;
            prod[0] = prod[0] ^ (prod[2] << 8);
            prod[0] ^= prod[0] >> 4;
            prod[0] &= 0x0f0f0f0f0f0f0f0f;

            row_a = a_4row[1] & 0xffff;
            a_4row[1] >>= 16;
            repeat = 0x0001000100010001*row_a;
            prod[1] = tb_4col & repeat;
            prod[1] ^= prod[1] >> 8;
            prod[1] &= 0x00ff00ff00ff00ff;
            row_a = a_4row[3] & 0xffff;
            a_4row[3] >>= 16;
            repeat = 0x0001000100010001*row_a;
            prod[3] = tb_4col & repeat;
            prod[3] ^= prod[3] >> 8;
            prod[3] &= 0x00ff00ff00ff00ff;
            prod[1] = prod[1] ^ (prod[3] << 8);
            prod[1] ^= prod[1] >> 4;
            prod[1] &= 0x0f0f0f0f0f0f0f0f;

            prod[0] = prod[0] ^ (prod[1] << 4);
            prod[0] ^= prod[0] << 2;
            prod[0] ^= prod[0] << 1;
            prod[0] &= 0x8888888888888888;
            
            out >>= 1;
            out |= prod[0];
        }
        out4x16[i_4col] = out;
    }
}

__m256i inline tbm_mult_t16x16_m256i(__m256i tb16x16, uint16_t a1x16[16])
{
    __m256i prod[8];
    for (uint8_t i_row = 0; i_row < 8; i_row++)
    {
        __m256i prodl = _mm256_and_si256(
            tb16x16, _mm256_set1_epi16(a1x16[i_row]));
        // prod0 ^= prod0 << 8; prod0 >>= 8;
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

// mult two 32x32 bit matrices with the second matrix transposed
// note: this code output is transposed, thus input were swapped...
void inline tbm_mult_t32x32_uint64(
    uint64_t tb2x32[16], uint64_t a2x32[16], uint64_t out2x32[16])
{
    for (uint8_t i_2col = 0; i_2col < 16; i_2col++)
    { 
        uint64_t out = 0;
        uint64_t tb_2col = tb2x32[i_2col];
        uint64_t prod[8];
        for (uint8_t i_bit = 0; i_bit < 2; i_bit++)
        {
            for (uint8_t i_row = 0; i_row < 8; i_row++)
            {
                uint64_t row_a = a2x32[i_row] >> (32*i_bit);
                row_a &= 0xffffffff;
                uint64_t repeat = 0x0000000100000001*row_a;
                uint64_t prodl = tb_2col & repeat;
                prodl ^= prodl >> 16;
                prodl &= 0x0000ffff0000ffff;
                
                row_a = a2x32[i_row+8] >> (32*i_bit);
                row_a &= 0xffffffff;
                repeat = 0x0000000100000001*row_a;
                uint64_t prodh = tb_2col & repeat;
                prodh ^= prodh << 16;
                prodh &= 0xffff0000ffff0000;
                
                prod[i_row] = prodl ^ prodh;
            }
            for (uint8_t i_row = 0; i_row < 4; i_row++)
            {
                uint64_t prodl = prod[i_row];
                prodl ^= prodl >> 8;
                prodl &= 0x00ff00ff00ff00ff;
                uint64_t prodh = prod[i_row+4];
                prodh ^= prodh << 8;
                prodh &= 0xff00ff00ff00ff00;
                prod[i_row] = prodl ^ prodh;
            }
            for (uint8_t i_row = 0; i_row < 2; i_row++)
            {
                uint64_t prodl = prod[i_row];
                prodl ^= prodl >> 4;
                prodl &= 0x0f0f0f0f0f0f0f0f;
                uint64_t prodh = prod[i_row+2];
                prodh ^= prodh << 4;
                prodh &= 0xf0f0f0f0f0f0f0f0;
                prod[i_row] = prodl ^ prodh;
            }
            for (uint8_t i_row = 0; i_row < 1; i_row++)
            {
                uint64_t prodl = prod[i_row];
                prodl ^= prodl >> 2;
                prodl &= 0x3333333333333333;
                uint64_t prodh = prod[i_row+1];
                prodh ^= prodh << 2;
                prodh &= 0xcccccccccccccccc;
                prod[i_row] = prodl ^ prodh;
            }

            prod[0] ^= prod[0] << 1;
            prod[0] &= 0xaaaaaaaaaaaaaaaa;
            
            out >>= 1;
            out |= prod[0];
        }
        out2x32[i_2col] = out;
    }
}

void tbm_mult_t8x8(
    uint64_t *in8x8, uint64_t *tb8x8, uint64_t n_mat, uint64_t *out8x8)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
#if defined(USE_AVX2)
        out8x8[i_mat] = tbm_mult_t8x8_m256i(in8x8[i_mat], tb8x8[i_mat]);
#else
        out8x8[i_mat] = tbm_mult_t8x8_uint64(in8x8[i_mat], tb8x8[i_mat]);
#endif
    }
}

void tbm_mult_t16x16(
    uint64_t *in4x16, uint64_t *tb4x16, uint64_t n_mat, uint64_t *out4x16)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
#if defined(USE_AVX2)
        __m256i in16x16 = _mm256_loadu_si256((__m256i *)in4x16);
        __m256i out16x16 = tbm_mult_t16x16_m256i(in16x16, (uint16_t *)tb4x16);
        _mm256_storeu_si256((__m256i *)out4x16, out16x16);
#else           
        tbm_mult_t16x16_uint64(in4x16, tb4x16, out4x16);
#endif
        in4x16 += 4;
        tb4x16 += 4;
        out4x16 += 4;
    }
}

void tbm_mult_t32x32(
    uint64_t *in2x32, uint64_t *tb2x32, uint64_t n_mat, uint64_t *out2x32)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
#if defined(USE_AVX2) && 0
        __m256i in8x32[4];
        for (uint8_t i_8row = 0; i_8row < 4; i_8row++)
            in8x32[i_8row] = _mm256_loadu_si256(((__m256i *)in2x32)+i_8row);
        tbm_transpose32x32_m256i(in8x32, in8x32);
        tbm_mult_t32x32_m256i(in8x32, (uint16_t *)tb2x32);
        for (uint8_t i_8row = 0; i_8row < 4; i_8row++)
            _mm256_storeu_si256(((__m256i *)out2x32)+i_8row, in8x32[i_8row]);
#else           
        tbm_mult_t32x32_uint64(in2x32, tb2x32, out2x32);
#endif
        in2x32 += 16;
        tb2x32 += 16;
        out2x32 += 16;
    }
}
