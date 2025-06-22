#include "tinybinmat_perm.h"
#include <string.h>

void tbm_eye_u64(uint32_t n_row_col, uint64_t *out)
{
    uint32_t n_dim_o = (n_row_col+7)/8; //!< number of rows/columns in octets
    uint8_t n_zero = 8*n_dim_o-n_row_col; //!< number of rows/columns to clear

    memset(out, 0, n_dim_o*n_dim_o*sizeof(uint64_t));
    for (uint32_t i_dim_o = 0; i_dim_o < n_dim_o; i_dim_o++)
    {
        out[i_dim_o*(n_dim_o+1)] = 0x0102040810204080ull;
    }

    // clear unused bits in the last columns
    uint64_t zero_mask = 0xffffffffffffffffull << (n_zero*8);
    out[n_dim_o*n_dim_o-1] &= zero_mask;
}
