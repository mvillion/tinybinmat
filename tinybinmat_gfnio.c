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

//______________________________________________________________________________
__m256i inline tbm_transpose8x8_m256i_gfni(__m256i in8x8_4)
{
    // _mm256_gf2p8affine_epi64_epi8(I, A, 0) is (A*I.T).T = A.T
    __m256i eye_8x8_4 = _mm256_set1_epi64x(0x0102040810204080);
    return _mm256_gf2p8affine_epi64_epi8(eye_8x8_4, in8x8_4, 0);
}

void tbm_transpose_gfnio_1d(uint64_t *in, uint64_t n8x8, uint64_t *out)
{
    uint64_t i8x8; //!< index for 4 8x8 blocks
    for (i8x8 = 0; i8x8 < n8x8/4*4; i8x8 += 4)
    {
        // load 4x8x8 blocks
        __m256i in8x8_4 = _mm256_loadu_si256((__m256i *)(in+i8x8));
        // transpose 4x8x8 blocks
        __m256i out8x8_4 = tbm_transpose8x8_m256i_gfni(in8x8_4);
        // store transposed 4x8x8 blocks
        _mm256_storeu_si256((__m256i *)(out+i8x8), out8x8_4);
    }
    if (i8x8 == n8x8)
        return; // all blocks are processed
    __m256i in8x8_4 = _mm256_loadu_si256((__m256i *)(in+i8x8));
    __m256i out8x8_4 = tbm_transpose8x8_m256i_gfni(in8x8_4);
    __m256i mask = _mm256_set_epi64x(3, 2, 1, 0); //!< mask for the last block
    mask = _mm256_cmpgt_epi64(_mm256_set1_epi64x(n8x8-i8x8), mask);
    _mm256_maskstore_epi64((long long int *)(out+i8x8), mask, out8x8_4);
}

void tbm_transpose_gfnio_2x2(uint64_t *in, uint64_t n_mat, uint64_t *out)
{
    uint64_t i8x8; //!< index for 4 8x8 blocks
    for (i8x8 = 0; i8x8 < n_mat*4; i8x8 += 4)
    {
        // load 4x8x8 blocks
        __m256i in8x8_4 = _mm256_loadu_si256((__m256i *)(in+i8x8));
        // transpose 4x8x8 blocks
        __m256i a3210r = tbm_transpose8x8_m256i_gfni(in8x8_4);
        __m256i a3120r = _mm256_permute4x64_epi64(
            a3210r, _MM_SHUFFLE(3, 1, 2, 0));
    
        // store transposed 4x8x8 blocks
        _mm256_storeu_si256((__m256i *)(out+i8x8), a3120r);
    }
}

#pragma GCC push_options //----------------------------------------------------
#pragma GCC optimize("no-tree-vectorize")
void tbm_transpose_gfnio(
    uint64_t *in, uint64_t n_mat, uint32_t n_row8, uint32_t n_col8, 
    uint64_t *out)
{
    if (n_row8 == 1 || n_col8 == 1)
    {
        // no block transpose needed
        uint64_t n8x8 = n_mat*n_row8*n_col8; //!< number of 8x8 blocks
        return tbm_transpose_gfnio_1d(in, n8x8, out);
    }
    else if (n_row8 == 2 && n_col8 == 2)
    {
        return tbm_transpose_gfnio_2x2(in, n_mat, out);
    }

    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        uint64_t *in_mat = in + i_mat*n_row8*n_col8;
        uint64_t *out_mat = out + i_mat*n_col8*n_row8;
        for (uint32_t i_row = 0; i_row < n_row8; i_row++)
            for (uint32_t i_col = 0; i_col < n_col8; i_col++)
                out_mat[i_col*n_row8+i_row] = in_mat[i_row*n_col8+i_col];
    }
    uint64_t i8x8; //!< index for 4 8x8 blocks
    uint64_t n8x8 = n_mat*n_row8*n_col8; //!< number of 8x8 blocks
    for (i8x8 = 0; i8x8 < n8x8/4*4; i8x8 += 4)
    {
        // load 4x8x8 blocks
        __m256i in8x8_4 = _mm256_loadu_si256((__m256i *)(out+i8x8));
        // transpose 4x8x8 blocks
        __m256i out8x8_4 = tbm_transpose8x8_m256i_gfni(in8x8_4);
        // store transposed 4x8x8 blocks
        _mm256_storeu_si256((__m256i *)(out+i8x8), out8x8_4);
    }
    if (i8x8 == n8x8)
        return; // all blocks are processed
    __m256i in8x8_4 = _mm256_loadu_si256((__m256i *)(out+i8x8));
    __m256i out8x8_4 = tbm_transpose8x8_m256i_gfni(in8x8_4);
    __m256i mask = _mm256_set_epi64x(3, 2, 1, 0); //!< mask for the last block
    mask = _mm256_cmpgt_epi64(_mm256_set1_epi64x(n8x8-i8x8), mask);
    _mm256_maskstore_epi64((long long int *)(out+i8x8), mask, out8x8_4);
}
#pragma GCC pop_options //-----------------------------------------------------


