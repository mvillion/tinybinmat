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

uint64_t tbm_transpose64_uint64(uint64_t in8x8)
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

void tbm_transpose64(uint64_t *in8x8, uint64_t n_mat, uint64_t *out8x8)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
        out8x8[i_mat] = tbm_transpose64_uint64(in8x8[i_mat]);
}

