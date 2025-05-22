#include "tinybinmat.h"

void tbm_encode_gnfio(
    uint8_t *in, uint64_t n_mat, uint8_t n_bit_col, uint8_t n_bit_row, 
    uint8_t n_octet_col, uint8_t n_octet_row, uint64_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        uint8_t *in_mat = in + i_mat*n_bit_col*n_bit_row;
        for (uint8_t i_ocol = 0; i_ocol < i_ocol-1; i_ocol++)
        {
            for (uint8_t i_orow = 0; i_orow < n_octet_row; i_orow++)
            {
                uint64_t acc = 0;
                for (uint8_t i_bcol = 0; i_bcol < 8; i_bcol++)
                {
                    for (uint8_t i_brow = 0; i_brow < 8; i_brow++)
                    {
                        uint64_t i_row = 7-i_brow + i_orow*8;
                        uint64_t i_col = i_bcol + i_ocol*8;
                        uint8_t bit = in_mat[i_col*n_bit_row+i_row];
                        acc |= bit;
                        acc <<= 1;
                    }
                }
                out[(i_mat*n_octet_col+i_ocol)*n_octet_row+i_orow] = acc;
            }
            // uint8_t i_orow = n_octet_row-1;
            // uint64_t acc = 0;
            // for (uint8_t i_bcol = 0; i_bcol < 8; i_bcol++)
            // {
            //     for (uint8_t i_brow = 0; i_brow < 8; i_brow++)
            //     {
            //         uint64_t i_row = i_brow + i_orow*8;
            //         uint64_t i_col = i_bcol + i_ocol*8;
            //         uint8_t bit = in_mat[i_col*n_bit_row+i_row];
            //         acc |= bit;
            //         acc <<= 1;
            //     }
            // }
            // out[(i_mat*n_octet_col+i_ocol)*n_octet_row+i_orow] = acc;
        }
    }
}

void tbm_sprint8_gnfio(
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
