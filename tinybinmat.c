#include "tinybinmat.h"

void print_avx2_uint64(__m256i reg)
{
    uint64_t *ptr = (uint64_t *)&reg;
    for (uint8_t k = 3; k != 0; k--)
        printf("%016lx ", ptr[k]);
    printf("%016lx\n", ptr[0]);
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

void tbm_encode8(
    uint8_t *in, uint64_t n_mat, uint8_t n_bit, uint8_t n_bit_raw, 
    uint8_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        for (uint8_t i_bit_raw = 0; i_bit_raw < n_bit; i_bit_raw++)
        {
            uint8_t acc = 0;
            for (uint8_t i_bit = 0; i_bit < n_bit; i_bit++)
            {
                uint8_t bit = in[(i_mat*n_bit+i_bit_raw)*n_bit+i_bit];
                acc |= (bit & 1) << i_bit;
            }
            out[i_mat*n_bit_raw+i_bit_raw] = acc;
        }
        for (uint8_t i_bit_raw = n_bit; i_bit_raw < n_bit_raw; i_bit_raw++)
            out[i_mat*n_bit_raw+i_bit_raw] = 0;
    }
}

void tbm_encode16(
    uint8_t *in, uint64_t n_mat, uint8_t n_bit, uint8_t n_bit_raw, 
    uint16_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        for (uint8_t i_bit_raw = 0; i_bit_raw < n_bit; i_bit_raw++)
        {
            uint16_t acc = 0;
            for (uint8_t i_bit = 0; i_bit < n_bit; i_bit++)
            {
                uint8_t bit = in[(i_mat*n_bit+i_bit_raw)*n_bit+i_bit];
                acc |= (bit & 1) << i_bit;
            }
            out[i_mat*n_bit_raw+i_bit_raw] = acc;
        }
        for (uint8_t i_bit_raw = n_bit; i_bit_raw < n_bit_raw; i_bit_raw++)
            out[i_mat*n_bit_raw+i_bit_raw] = 0;
    }
}

void tbm_encode32(
    uint8_t *in, uint64_t n_mat, uint8_t n_bit, uint8_t n_bit_raw, 
    uint32_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        for (uint8_t i_bit_raw = 0; i_bit_raw < n_bit; i_bit_raw++)
        {
            uint32_t acc = 0;
            for (uint8_t i_bit = 0; i_bit < n_bit; i_bit++)
            {
                uint8_t bit = in[(i_mat*n_bit+i_bit_raw)*n_bit+i_bit];
                acc |= (bit & 1) << i_bit;
            }
            out[i_mat*n_bit_raw+i_bit_raw] = acc;
        }
        for (uint8_t i_bit_raw = n_bit; i_bit_raw < n_bit_raw; i_bit_raw++)
            out[i_mat*n_bit_raw+i_bit_raw] = 0;
    }
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

//______________________________________________________________________________
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

__m256i inline tbm_transpose8x8_m256i_gfni(__m256i in8x8_4)
{
    // _mm256_gf2p8affine_epi64_epi8(I, A, 0) is (A*I.T).T = A.T
    // a flipud of the matrix is needed before and after the transformation
    // as conventions are different
    __m128i reverse8_2col = _mm_set_epi8(
        8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7);
    __m256i reverse8_col = _mm256_set_m128i(reverse8_2col, reverse8_2col);
    __m256i in8x8_4rev = _mm256_shuffle_epi8(in8x8_4, reverse8_col);

    __m256i eye_8x8_4 = _mm256_set1_epi64x(0x0102040810204080);
    return _mm256_shuffle_epi8(
        _mm256_gf2p8affine_epi64_epi8(eye_8x8_4, in8x8_4rev, 0), reverse8_col);
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
void tbm_transpose8x8(uint8_t *in, uint64_t n_mat, uint8_t *out)
{
    uint64_t i_avx2 = 0;
#if defined(USE_AVX2)
    i_avx2 = n_mat/4*4;
    for (uint64_t i_mat = 0; i_mat < i_avx2; i_mat += 4)
    {
        __m256i in8x8_4 = _mm256_loadu_si256((__m256i *)in);
        __m256i out8x8_4 = tbm_transpose8x8_m256i_gfni(in8x8_4);
        _mm256_storeu_si256((__m256i *)out, out8x8_4);
        in += 8*4;
        out += 8*4;
    }
#endif
    for (uint64_t i_mat = i_avx2; i_mat < n_mat; i_mat++)
    {
        *((uint64_t *)out) = tbm_transpose8x8_uint64(*((uint64_t *)in));
        in += 8;
        out += 8;
    }
}

void tbm_transpose16x16(uint16_t *in, uint64_t n_mat, uint16_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
#if defined(USE_AVX2)
        __m256i in16x16 = _mm256_loadu_si256((__m256i *)in);
        __m256i out16x16 = tbm_transpose16x16_m256i(in16x16);
        _mm256_storeu_si256((__m256i *)out, out16x16);
#else
        uint64_t *in4x16 = (uint64_t *)in;
        uint64_t *out4x16 = (uint64_t *)out;
        tbm_transpose16x16_uint64(
            in4x16[0], in4x16[1], in4x16[2], in4x16[3],
            out4x16+0, out4x16+1, out4x16+2, out4x16+3);
#endif
        in += 16;
        out += 16;
    }
}

void tbm_transpose32x32(uint32_t *in, uint64_t n_mat, uint32_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
#if defined(USE_AVX2)
        __m256i in8x32[4];
        for (uint8_t i_8row = 0; i_8row < 4; i_8row++)
            in8x32[i_8row] = _mm256_loadu_si256(((__m256i *)in)+i_8row);
        tbm_transpose32x32_m256i(in8x32, in8x32);
        for (uint8_t i_8row = 0; i_8row < 4; i_8row++)
            _mm256_storeu_si256(((__m256i *)out)+i_8row, in8x32[i_8row]);
#else
        tbm_transpose32x32_uint64((uint64_t *)in, (uint64_t *)out);
#endif
        in += 32;
        out += 32;
    }
}
#pragma GCC pop_options //-----------------------------------------------------

//______________________________________________________________________________
// multiply two 8x8 bit matrices
uint64_t inline tbm_mult8x8_uint64(uint64_t a, uint8_t b[8])
{
    uint64_t out = 0;
    for (uint8_t i_bit = 0; i_bit < 8; i_bit++)
    {
        // create bit mask from the least significant bits in a
        uint64_t bit_a = a & 0x0101010101010101;
        bit_a *= 0xff;
        a >>= 1;
        uint64_t prod = bit_a & (0x0101010101010101*b[i_bit]);
        out ^= prod;
    }
    return out;
}

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

__m256i inline tbm_mult8x8_m256i_gfni(__m256i a, __m256i b)
{
    // _mm256_gf2p8affine_epi64_epi8(B, A, 0) is (A*B.T).T
    // _mm256_gf2p8affine_epi64_epi8(A, B.T, 0) is (B.T*A.T).T = A*B
    // the second form needs a single transposition
    // J*_mm256_gf2p8affine_epi64_epi8(J*A, (J*B).T, 0) is A*(J*B)
    // a flipud of the matrix is needed before and after the transformation
    // as conventions are different
    __m128i reverse8_2col = _mm_set_epi8(
        8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7);

    __m256i reverse8_col = _mm256_set_m128i(reverse8_2col, reverse8_2col);
    __m256i b8x8_4rev = _mm256_shuffle_epi8(b, reverse8_col);

    __m256i eye_8x8_4 = _mm256_set1_epi64x(0x0102040810204080);
    __m256i b8x8_4t = _mm256_gf2p8affine_epi64_epi8(eye_8x8_4, b8x8_4rev, 0);
    return _mm256_gf2p8affine_epi64_epi8(a, b8x8_4t, 0);
}

// multiply two 16x16 bit matrices
void inline tbm_mult16x16_uint64(uint64_t a[4], uint16_t b[16], uint64_t out[4])
{
    uint64_t a_bck[4];
    for (uint8_t i_4row = 0; i_4row < 4; i_4row++)
    {
        a_bck[i_4row] = a[i_4row];
    }
    for (uint8_t i_4row = 0; i_4row < 4; i_4row++)
    {
        out[i_4row] = 0;
        for (uint8_t i_bit = 0; i_bit < 16; i_bit++)
        {
            // create bit mask from the least significant bits in a
            uint64_t bit_a = a_bck[i_4row] & 0x0001000100010001;
            bit_a *= 0xffff;
            a_bck[i_4row] >>= 1;
            uint64_t prod = bit_a & (0x0001000100010001*b[i_bit]);
            out[i_4row] ^= prod;
        }
    }
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

__m256i inline tbm_mult16x16_m256i_gfni(__m256i a, __m256i b)
{
    // We want to convert the 16x16 matrix to four 8x8 matrix
    // following the order: [[sub1, sub0], [sub3, sub2]]
    // first 16 bits are in least signicant bir order: b15 down to b0
    // odd and even octets represent submatrices sub1 and sub0
    // but as 16x16 matrix encodes a row as [b0, b1, ... b15]
    // matrix sub0 is actually located in the odd octets
    __m128i split8x8_2 = _mm_set_epi8(
        14, 12, 10, 8, 6, 4, 2, 0, 15, 13, 11, 9, 7, 5, 3, 1);
    __m256i split8x8_4 = _mm256_set_m128i(split8x8_2, split8x8_2);
    __m256i a_3210 = _mm256_shuffle_epi8(a, split8x8_4);
    __m256i b_3210 = _mm256_shuffle_epi8(b, split8x8_4);

    __m256i a_3311 = _mm256_permute4x64_epi64(a_3210, _MM_SHUFFLE(3, 3, 1, 1));
    __m256i a_2200 = _mm256_permute4x64_epi64(a_3210, _MM_SHUFFLE(2, 2, 0, 0));
    __m256i b_1010 = _mm256_permute4x64_epi64(b_3210, _MM_SHUFFLE(1, 0, 1, 0));
    __m256i b_3232 = _mm256_permute4x64_epi64(b_3210, _MM_SHUFFLE(3, 2, 3, 2));

    __m256i out = _mm256_xor_si256(
        tbm_mult8x8_m256i_gfni(a_3311, b_1010),
        tbm_mult8x8_m256i_gfni(a_2200, b_3232));
    
    __m128i unsplit8x8_2 = _mm_set_epi8(
        7, 15, 6, 14, 5, 13, 4, 12, 3, 11, 2, 10, 1, 9, 0, 8);
    __m256i unsplit8x8_4 = _mm256_set_m128i(unsplit8x8_2, unsplit8x8_2);
    return _mm256_shuffle_epi8(out, unsplit8x8_4);
}

// multiply two 32x32 bit matrices
void inline tbm_mult32x32_uint64(
    uint64_t a[16], uint32_t b[32], uint64_t out[16])
{
    uint64_t a_bck[16];
    for (uint8_t i_2row = 0; i_2row < 16; i_2row++)
    {
        a_bck[i_2row] = a[i_2row];
    }
    for (uint8_t i_2row = 0; i_2row < 16; i_2row++)
    {
        out[i_2row] = 0;
        for (uint8_t i_bit = 0; i_bit < 32; i_bit++)
        {
            // create bit mask from the least significant bits in a
            uint64_t bit_a = a_bck[i_2row] & 0x0000000100000001;
            bit_a *= 0xffffffff;
            a_bck[i_2row] >>= 1;
            uint64_t prod = bit_a & (0x0000000100000001*b[i_bit]);
            out[i_2row] ^= prod;
        }
    }
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
void tbm_mult8x8(uint8_t *in, uint8_t *in2, uint64_t n_mat, uint8_t *out)
{
    uint64_t i_avx2 = 0;
#if defined(USE_AVX2)
    i_avx2 = n_mat/4*4;
    for (uint64_t i_mat = 0; i_mat < i_avx2; i_mat += 4)
    {
        __m256i in8x8_4 = _mm256_loadu_si256((__m256i *)in);
        __m256i in2_8x8_4 = _mm256_loadu_si256((__m256i *)in2);
        __m256i out8x8_4 = tbm_mult8x8_m256i_gfni(in8x8_4, in2_8x8_4);
        _mm256_storeu_si256((__m256i *)out, out8x8_4);
        in += 8*4;
        in2 += 8*4;
        out += 8*4;
    }
#endif
    for (uint64_t i_mat = i_avx2; i_mat < n_mat; i_mat++)
    {
        *((uint64_t *)out) = tbm_mult8x8_uint64(*((uint64_t *)in), in2);
        in += 8;
        in2 += 8;
        out += 8;
    }
}

void tbm_mult16x16(uint16_t *in, uint16_t *in2, uint64_t n_mat, uint16_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
#if defined(USE_AVX2)
        __m256i in16x16 = _mm256_loadu_si256((__m256i *)in);
        __m256i in2_16x16 = _mm256_loadu_si256((__m256i *)in2);
        __m256i out16x16 = tbm_mult16x16_m256i_gfni(in16x16, in2_16x16);
        // __m256i out16x16 = tbm_mult16x16_m256i(in16x16, in2);
        _mm256_storeu_si256((__m256i *)out, out16x16);
#else           
        tbm_mult16x16_uint64((uint64_t *)in, in2, (uint64_t *)out);
#endif
        in += 16;
        in2 += 16;
        out += 16;
    }
}

void tbm_mult32x32(uint32_t *in, uint32_t *in2, uint64_t n_mat, uint32_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
#if defined(USE_AVX2)
        __m256i in8x32[4];
        __m256i out8x32[4];
        for (uint8_t i_8row = 0; i_8row < 4; i_8row++)
            in8x32[i_8row] = _mm256_loadu_si256(((__m256i *)in)+i_8row);
        tbm_mult32x32_m256i(in8x32, in2, out8x32);
        for (uint8_t i_8row = 0; i_8row < 4; i_8row++)
            _mm256_storeu_si256(((__m256i *)out)+i_8row, out8x32[i_8row]);
#else           
        tbm_mult32x32_uint64((uint64_t *)in, in2, (uint64_t *)out);
#endif
        in += 32;
        in2 += 32;
        out += 32;
    }
}
#pragma GCC pop_options //-----------------------------------------------------

//______________________________________________________________________________
// multiply two 8x8 bit matrices with the second matrix transposed
uint64_t inline tbm_mult_t8x8_uint64(uint64_t a8x8, uint8_t tb[8])
{
    uint64_t out = 0;
    for (uint8_t i_bit = 0; i_bit < 8; i_bit++)
    {
        uint64_t repeat = 0x0101010101010101*tb[i_bit];
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

// multiply 4 groups of two 8x8 bit matrices with the second matrces transposed
__m256i inline tbm_mult_t8x8_m256i_gfni(__m256i a, __m256i b)
{
    // _mm256_gf2p8affine_epi64_epi8(B, A, 0) is (A*B.T).T
    // _mm256_gf2p8affine_epi64_epi8(A, B, 0) is (B*A.T).T = A*B.T
    // J*_mm256_gf2p8affine_epi64_epi8(J*A, J*B, 0) is A*B.T*J.T
    // a flipud of the matrix is is only needed on the second matrix
    __m128i reverse8_2col = _mm_set_epi8(
        8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7);
    __m256i reverse8_col = _mm256_set_m128i(reverse8_2col, reverse8_2col);
    __m256i b8x8_4rev = _mm256_shuffle_epi8(b, reverse8_col);

    return _mm256_gf2p8affine_epi64_epi8(a, b8x8_4rev, 0);
}

// multiply two 16x16 bit matrices with the second matrix transposed
// note: this code output is transposed, thus input were swapped...
void inline tbm_mult_t16x16_uint64(
    uint64_t tb4x16[4], uint16_t a[16], uint64_t out4x16[4])
{
    for (uint8_t i_4col = 0; i_4col < 4; i_4col++)
    { 
        uint64_t out = 0;
        uint64_t tb_4col = tb4x16[i_4col];
        uint64_t prod[4];
        for (uint8_t i_bit = 0; i_bit < 4; i_bit++)
        {
            uint64_t repeat = 0x0001000100010001*a[4*0+i_bit];
            prod[0] = tb_4col & repeat;
            prod[0] ^= prod[0] >> 8;
            prod[0] &= 0x00ff00ff00ff00ff;
            repeat = 0x0001000100010001*a[4*2+i_bit];
            prod[2] = tb_4col & repeat;
            prod[2] ^= prod[2] >> 8;
            prod[2] &= 0x00ff00ff00ff00ff;
            prod[0] = prod[0] ^ (prod[2] << 8);
            prod[0] ^= prod[0] >> 4;
            prod[0] &= 0x0f0f0f0f0f0f0f0f;

            repeat = 0x0001000100010001*a[4*1+i_bit];
            prod[1] = tb_4col & repeat;
            prod[1] ^= prod[1] >> 8;
            prod[1] &= 0x00ff00ff00ff00ff;
            repeat = 0x0001000100010001*a[4*3+i_bit];
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

// multiply two 32x32 bit matrices with the second matrix transposed
// note: this code output is transposed, thus input were swapped...
void inline tbm_mult_t32x32_uint64(
    uint64_t tb2x32[16], uint32_t a1x32[32], uint64_t out2x32[16])
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
                uint64_t repeat = 0x0000000100000001*a1x32[2*i_row+i_bit];
                uint64_t prodl = tb_2col & repeat;
                prodl ^= prodl >> 16;
                prodl &= 0x0000ffff0000ffff;
                
                repeat = 0x0000000100000001*a1x32[16+2*i_row+i_bit];
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
void tbm_mult_t8x8(uint8_t *in, uint8_t *in2t, uint64_t n_mat, uint8_t *out)
{
    uint64_t i_avx2 = 0;
#if defined(USE_AVX2)
    i_avx2 = n_mat/4*4;
    for (uint64_t i_mat = 0; i_mat < i_avx2; i_mat += 4)
    {
        __m256i in8x8_4 = _mm256_loadu_si256((__m256i *)in);
        __m256i in2t_8x8_4 = _mm256_loadu_si256((__m256i *)in2t);
        __m256i out8x8_4 = tbm_mult_t8x8_m256i_gfni(in8x8_4, in2t_8x8_4);
        _mm256_storeu_si256((__m256i *)out, out8x8_4);
        in += 8*4;
        in2t += 8*4;
        out += 8*4;
    }
    *((uint64_t *)out) = tbm_mult_t8x8_single_m256i(
            *((uint64_t *)in), *((uint64_t *)in2t));
#endif
    for (uint64_t i_mat = i_avx2; i_mat < n_mat; i_mat++)
    {
        *((uint64_t *)out) = tbm_mult_t8x8_uint64(*((uint64_t *)in), in2t);
        in += 8;
        in2t += 8;
        out += 8;
    }
}

void tbm_mult_t16x16(
    uint16_t *in, uint16_t *in2t, uint64_t n_mat, uint16_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
#if defined(USE_AVX2)
        __m256i in16x16 = _mm256_loadu_si256((__m256i *)in);
        __m256i out16x16 = tbm_mult_t16x16_m256i(in16x16, in2t);
        _mm256_storeu_si256((__m256i *)out, out16x16);
#else           
        tbm_mult_t16x16_uint64((uint64_t *)in, in2t, (uint64_t *)out);
#endif
        in += 16;
        in2t += 16;
        out += 16;
    }
}

void tbm_mult_t32x32(
    uint32_t *in, uint32_t *in2t, uint64_t n_mat, uint32_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
#if defined(USE_AVX2)
        __m256i in8x32[4];
        for (uint8_t i_8row = 0; i_8row < 4; i_8row++)
            in8x32[i_8row] = _mm256_loadu_si256(((__m256i *)in)+i_8row);
        tbm_mult_t32x32_m256i(in8x32, in2t);
        for (uint8_t i_8row = 0; i_8row < 4; i_8row++)
            _mm256_storeu_si256(((__m256i *)out)+i_8row, in8x32[i_8row]);
#else           
        tbm_mult_t32x32_uint64((uint64_t *)in, in2t, (uint64_t *)out);
#endif
        in += 32;
        in2t += 32;
        out += 32;
}
}
#pragma GCC pop_options //-----------------------------------------------------
