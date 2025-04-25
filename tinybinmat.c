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

uint64_t tbm_transpose64(uint64_t in8x8)
{
    // input is 8x8 bit matrix with rows 8 rows: 0x0706050403020100
    // 1st bit in rows is LSB, thus reversed compared to matrix notation
    uint64_t ur_mask8x8 = 0x00000000f0f0f0f0 ; // up right 4x4 bits
    // uint64_t = dl_mask8x8 = 0x0f0f0f0f00000000; // down left 4x4 bits
    // dl_mask8x8 == ur_mask8x8 << (32-4)
    uint64_t xor = in8x8 ^ (in8x8 >> 28);
    xor &= ur_mask8x8;
    in8x8 ^= xor;
    xor <<= 28;
    in8x8 ^= xor;
    uint64_t ur_mask4x4 = 0x0000cccc0000cccc ; // 4 up right 2x2 bits
    // uint64_t = dl_mask4x4 = 0x3333000033330000; // 4 down left 2x2 bits
    // dl_mask8x8 == ur_mask8x8 << (16-2)
    xor = in8x8 ^ (in8x8 >> 14);
    xor &= ur_mask4x4;
    in8x8 ^= xor;
    xor <<= 14;
    in8x8 ^= xor;
    uint64_t ur_mask2x2 = 0x00aa00aa00aa00aa ; // 16 up right 1x1 bits
    // uint64_t = dl_mask4x4 = 0x5500550055005500; // 16 down left 1x1 bits
    // dl_mask8x8 == ur_mask8x8 << (8-1)
    xor = in8x8 ^ (in8x8 >> 7);
    xor &= ur_mask2x2;
    in8x8 ^= xor;
    xor <<= 7;
    in8x8 ^= xor;
    return in8x8;
}

void inline transpose8x8_int16(
    __m256i hgfedcba4_hgfedcba0, __m256i hgfedcba5_hgfedcba1,
    __m256i hgfedcba6_hgfedcba2, __m256i hgfedcba7_hgfedcba3,
    __m256i *b76543210_a76543210, __m256i *d76543210_c76543210,
    __m256i *f76543210_e76543210, __m256i *h76543210_g76543210)
{
    __m256i d5d4c5c4b5b4a5a4_d1d0c1c0b1b0a1a0 = _mm256_unpacklo_epi16(
        hgfedcba4_hgfedcba0, hgfedcba5_hgfedcba1);
    __m256i h5h4g5g4f5f4e5e4_h1h0g1g0f1f0e1e0 = _mm256_unpackhi_epi16(
        hgfedcba4_hgfedcba0, hgfedcba5_hgfedcba1);
    __m256i d7d6c7c6b7b6a7a6_d3d2c3c2b3b2a3a2 = _mm256_unpacklo_epi16(
        hgfedcba6_hgfedcba2, hgfedcba7_hgfedcba3);
    __m256i h7h6g7g6f7f6e7e6_h3h2g3g2f3f2e3e2 = _mm256_unpackhi_epi16(
        hgfedcba6_hgfedcba2, hgfedcba7_hgfedcba3);

    __m256i b7b6b5b4a7a6a5a4_b3b2b1b0a3a2a1a0 = _mm256_unpacklo_epi32(
        d5d4c5c4b5b4a5a4_d1d0c1c0b1b0a1a0, d7d6c7c6b7b6a7a6_d3d2c3c2b3b2a3a2);
    __m256i d7d6d5d4c7c6c5c4_d3d2d1d0c3c2c1c0 = _mm256_unpackhi_epi32(
        d5d4c5c4b5b4a5a4_d1d0c1c0b1b0a1a0, d7d6c7c6b7b6a7a6_d3d2c3c2b3b2a3a2);
    __m256i f7f6f5f4e7e6e5e4_f3f2f1f0e3e2e1e0 = _mm256_unpacklo_epi32(
        h5h4g5g4f5f4e5e4_h1h0g1g0f1f0e1e0, h7h6g7g6f7f6e7e6_h3h2g3g2f3f2e3e2);
    __m256i h7h6h5h4g7g6g5g4_h3h2h1h0g3g2g1g0 = _mm256_unpackhi_epi32(
        h5h4g5g4f5f4e5e4_h1h0g1g0f1f0e1e0, h7h6g7g6f7f6e7e6_h3h2g3g2f3f2e3e2);

    *b76543210_a76543210 = _mm256_permute4x64_epi64(
        b7b6b5b4a7a6a5a4_b3b2b1b0a3a2a1a0, _MM_SHUFFLE(3, 1, 2, 0));
    *d76543210_c76543210 = _mm256_permute4x64_epi64(
        d7d6d5d4c7c6c5c4_d3d2d1d0c3c2c1c0, _MM_SHUFFLE(3, 1, 2, 0));
    *f76543210_e76543210 = _mm256_permute4x64_epi64(
        f7f6f5f4e7e6e5e4_f3f2f1f0e3e2e1e0, _MM_SHUFFLE(3, 1, 2, 0));
    *h76543210_g76543210 = _mm256_permute4x64_epi64(
        h7h6h5h4g7g6g5g4_h3h2h1h0g3g2g1g0, _MM_SHUFFLE(3, 1, 2, 0));
}
