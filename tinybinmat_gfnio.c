#include "tinybinmat.h"
#include "tinybinmat_gfnio.h"

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
