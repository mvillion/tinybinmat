#include <byteswap.h>
#include <string.h>
#include "tinybinmat_conv.h"

void tbm_circul_u64_single(uint8_t *in, uint32_t n_row_col, uint64_t *out)
{
    uint32_t n_dim_o = (n_row_col+7)/8; //!< number of rows/columns in octets
    uint8_t n_zero = 8*n_dim_o-n_row_col; //!< number of rows/columns to clear
    if (0 < n_zero)
    {
        // clear unused bits in the last columns
        uint8_t zero_mask = 0xff >> n_zero;
        in[n_dim_o-1] &= zero_mask;
    }

    // clear unused bits in the last columns
    uint64_t zero_mask = 0xff >> n_zero;
    zero_mask *= 0x0101010101010101; //!< mask for the last columns

    uint64_t lower_mask = 0x000103070f1f3f7fll; //!< upper matrix mask
    memset(out, 0, n_dim_o*n_dim_o*sizeof(uint64_t));
    uint64_t prev_lower = 0; //! previous lower matrix value
    for (uint32_t i_col_o = 0; i_col_o < n_dim_o; i_col_o++)
    {
        uint64_t near_circ = bswap_64(in[i_col_o]*0x8040201008040201ull);
        uint64_t circ = near_circ & ~lower_mask;
        circ |= prev_lower;
        prev_lower = near_circ & lower_mask;
        prev_lower <<= 8;
        prev_lower |= in[i_col_o] >> 1;
        out[i_col_o] = circ;
    }
    // finish diagonal
    prev_lower <<= n_zero;
    prev_lower |= (out[n_dim_o-1] & ~zero_mask) >> (8-n_zero);
    out[n_dim_o-1] &= zero_mask;
    out[0] |= prev_lower;
    for (uint32_t i_row_o = 1; i_row_o < n_dim_o-1; i_row_o++)
    {
        for (uint32_t i_col_o = 1; i_col_o < n_dim_o; i_col_o++)
            out[i_row_o*n_dim_o+i_col_o] = out[(i_row_o-1)*n_dim_o+i_col_o-1];
        out[(i_row_o+1)*n_dim_o-1] &= zero_mask;
        uint64_t prev = out[i_row_o*n_dim_o-1];
        prev <<= n_zero;
        prev |= (out[i_row_o*n_dim_o-2] & ~zero_mask) >> (8-n_zero);
        out[i_row_o*n_dim_o] = prev;
    }
    uint32_t i_row_o = n_dim_o-1;
    uint64_t zero_mask_row = 0xffffffffffffffffull << (n_zero*8);
    {
        for (uint32_t i_col_o = 1; i_col_o < n_dim_o; i_col_o++)
        {
            uint64_t prev = out[(i_row_o-1)*n_dim_o+i_col_o-1];
            out[i_row_o*n_dim_o+i_col_o] = prev & zero_mask_row;
        }
        out[n_dim_o*n_dim_o-1] &= zero_mask;
        if (n_dim_o < 2)
        {
            out[n_dim_o*n_dim_o-1] &= zero_mask_row;
            return;
        }
        uint64_t prev = out[i_row_o*n_dim_o-1];
        prev <<= n_zero;
        prev |= (out[i_row_o*n_dim_o-2] & ~zero_mask) >> (8-n_zero);
        out[i_row_o*n_dim_o] = prev & zero_mask_row;
    }
}

void tbm_circul_u64(uint8_t *in, uint32_t n_mat, uint32_t n_row_col, uint64_t *out)
{
    uint32_t n_dim_o = (n_row_col+7)/8; //!< number of rows/columns in octets
    for (uint32_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        tbm_circul_u64_single(in, n_row_col, out);
        in += n_dim_o;
        out += n_dim_o*n_dim_o;
    }
}
