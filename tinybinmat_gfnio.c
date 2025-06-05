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

void tbm_transpose_gfnio_2x2(__m256i *in, uint64_t n_mat, __m256i *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        // load 4x8x8 blocks
        __m256i in8x8_4 = _mm256_loadu_si256(in+i_mat);
        // transpose 4x8x8 blocks
        __m256i a3210r = tbm_transpose8x8_m256i_gfni(in8x8_4);
        __m256i a3120r = _mm256_permute4x64_epi64(
            a3210r, _MM_SHUFFLE(3, 1, 2, 0));
    
        // store transposed 4x8x8 blocks
        _mm256_storeu_si256(out+i_mat, a3120r);
    }
}

void tbm_transpose_gfnio_4x4(__m256i *in, uint64_t n_mat, __m256i *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        __m256i in3210 = _mm256_loadu_si256(in+i_mat*4+0);
        __m256i in7654 = _mm256_loadu_si256(in+i_mat*4+1);
        __m256i inba98 = _mm256_loadu_si256(in+i_mat*4+2);
        __m256i infedc = _mm256_loadu_si256(in+i_mat*4+3);
    
        __m256i in9810 = _mm256_permute2x128_si256(in3210, inba98, 0x20);
        __m256i indc54 = _mm256_permute2x128_si256(in7654, infedc, 0x20);
        __m256i inba32 = _mm256_permute2x128_si256(in3210, inba98, 0x31);
        __m256i infe76 = _mm256_permute2x128_si256(in7654, infedc, 0x31);
    
        __m256i inc840 = _mm256_unpacklo_epi64(in9810, indc54);
        __m256i ind951 = _mm256_unpackhi_epi64(in9810, indc54);
        __m256i inea62 = _mm256_unpacklo_epi64(inba32, infe76);
        __m256i infb73 = _mm256_unpackhi_epi64(inba32, infe76);

        _mm256_storeu_si256(out+i_mat*4+0, tbm_transpose8x8_m256i_gfni(inc840));
        _mm256_storeu_si256(out+i_mat*4+1, tbm_transpose8x8_m256i_gfni(ind951));
        _mm256_storeu_si256(out+i_mat*4+2, tbm_transpose8x8_m256i_gfni(inea62));
        _mm256_storeu_si256(out+i_mat*4+3, tbm_transpose8x8_m256i_gfni(infb73));
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
        return tbm_transpose_gfnio_2x2((__m256i *)in, n_mat, (__m256i *)out);
    }
    else if (n_row8 == 4 && n_col8 == 4)
    {
        return tbm_transpose_gfnio_4x4((__m256i *)in, n_mat, (__m256i *)out);
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

//______________________________________________________________________________
__m256i inline tbm_mult8x8_m256i_gfnio(__m256i a, __m256i b)
{
    // _mm256_gf2p8affine_epi64_epi8(B, A, 0) is (A*B.T).T
    // _mm256_gf2p8affine_epi64_epi8(A, B.T, 0) is (B.T*A.T).T = A*B
    return _mm256_gf2p8affine_epi64_epi8(a, tbm_transpose8x8_m256i_gfni(b), 0);
}

__m256i inline tbm_mult16x16_m256i_gfnio(__m256i a3210, __m256i b3210)
{
    // a is:    b.T is:
    // [0, 1]   [0, 2]   [0, 0]   [0, 2]   [1, 1]   [1, 3]
    // [2, 3] @ [1, 3] = [2, 2] * [0, 2] + [3, 3] * [1, 3]
    __m256i a3311 = _mm256_permute4x64_epi64(a3210, _MM_SHUFFLE(3, 3, 1, 1));
    __m256i a2200 = _mm256_permute4x64_epi64(a3210, _MM_SHUFFLE(2, 2, 0, 0));
    __m256i b3131 = _mm256_permute4x64_epi64(b3210, _MM_SHUFFLE(3, 1, 3, 1));
    __m256i b2020 = _mm256_permute4x64_epi64(b3210, _MM_SHUFFLE(2, 0, 2, 0));

    __m256i out = _mm256_xor_si256(
        _mm256_gf2p8affine_epi64_epi8(a3311, b3131, 0),
        _mm256_gf2p8affine_epi64_epi8(a2200, b2020, 0));

    return out;
}

void inline tbm_mult32x32_m256i_gfnio(__m256i a[4], __m256i b[4])
{
    __m256i b_3210 = b[0];
    __m256i b_7654 = b[1];
    __m256i b_ba98 = b[2];
    __m256i b_fedc = b[3];

    __m256i b_9810 = _mm256_permute2x128_si256(b_3210, b_ba98, 0x20);
    __m256i b_dc54 = _mm256_permute2x128_si256(b_7654, b_fedc, 0x20);
    __m256i b_ba32 = _mm256_permute2x128_si256(b_3210, b_ba98, 0x31);
    __m256i b_fe76 = _mm256_permute2x128_si256(b_7654, b_fedc, 0x31);

    __m256i b_c840 = _mm256_unpacklo_epi64(b_9810, b_dc54);
    __m256i b_d951 = _mm256_unpackhi_epi64(b_9810, b_dc54);
    __m256i b_ea62 = _mm256_unpacklo_epi64(b_ba32, b_fe76);
    __m256i b_fb73 = _mm256_unpackhi_epi64(b_ba32, b_fe76);

    uint64_t *a64 = (uint64_t *)a;
    for (uint8_t i_row = 0; i_row < 4; i_row++)
    {
        // __m256i a3210 = a[i_row];

        __m256i repeat; //<! current product of a cell 4 times  
        __m256i prod; //<! current product of a cell and b row
        __m256i out; //<! accumulated sum of the products
        out = _mm256_setzero_si256();
        // repeat = _mm256_permute4x64_epi64(a3210, _MM_SHUFFLE(0, 0, 0, 0));
        repeat = _mm256_set1_epi64x(*a64++);
        prod = _mm256_gf2p8affine_epi64_epi8(repeat, b_c840, 0);
        out = _mm256_xor_si256(out, prod);

        // repeat = _mm256_permute4x64_epi64(a3210, _MM_SHUFFLE(1, 1, 1, 1));
        repeat = _mm256_set1_epi64x(*a64++);
        prod = _mm256_gf2p8affine_epi64_epi8(repeat, b_d951, 0);
        out = _mm256_xor_si256(out, prod);

        // repeat = _mm256_permute4x64_epi64(a3210, _MM_SHUFFLE(2, 2, 2, 2));
        repeat = _mm256_set1_epi64x(*a64++);
        prod = _mm256_gf2p8affine_epi64_epi8(repeat, b_ea62, 0);
        out = _mm256_xor_si256(out, prod);

        // repeat = _mm256_permute4x64_epi64(a3210, _MM_SHUFFLE(3, 3, 3, 3));
        repeat = _mm256_set1_epi64x(*a64++);
        prod = _mm256_gf2p8affine_epi64_epi8(repeat, b_fb73, 0);
        out = _mm256_xor_si256(out, prod);

        a[i_row] = out;
    }
}

void inline tbm_mult_gfnio_256(
    uint64_t *in, uint64_t n_mat, uint32_t n_row8, uint32_t n_col8,
    uint64_t *in2, uint32_t n_col8_2, uint64_t *out)
{
    __m256i mask = _mm256_set_epi64x(3, 2, 1, 0); //!< mask for the last block
    mask = _mm256_cmpgt_epi64(_mm256_set1_epi64x(n_col8_2-n_col8_2/4*4), mask);
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        uint64_t *in_mat = in + i_mat*n_row8*n_col8;
        uint64_t *in2_mat = in2 + i_mat*n_col8*n_col8_2;
        uint64_t *out_mat = out + i_mat*n_row8*n_col8_2;
        uint32_t i_row; //!< index of output row
        __m256i repeat; //!< value repeated 4 times for multiplication
        for (i_row = 0; i_row < n_row8; i_row++)
        {
            uint32_t i_col; //!< index of output colum
            for (i_col = 0; i_col < n_col8_2/4*4; i_col += 4)
            {
                __m256i acc = _mm256_setzero_si256();
                for (uint32_t i_dot = 0; i_dot < n_col8; i_dot++)
                {
                    repeat = _mm256_set1_epi64x(in_mat[i_row*n_col8+i_dot]);
                    __m256i b_8x8_4 = _mm256_loadu_si256(
                        (__m256i *)(in2_mat+i_dot*n_col8_2+i_col));
                    acc = _mm256_xor_si256(
                        acc, tbm_mult8x8_m256i_gfnio(repeat, b_8x8_4));
                }
                _mm256_storeu_si256(
                    (__m256i *)(out_mat+i_row*n_col8_2+i_col), acc);
            }
            if (i_col == n_col8_2)
                continue; // all blocks are processed
            __m256i acc = _mm256_setzero_si256();
            for (uint32_t i_dot = 0; i_dot < n_col8; i_dot++)
            {
                repeat = _mm256_set1_epi64x(in_mat[i_row*n_col8+i_dot]);
                __m256i b_8x8_4 = _mm256_loadu_si256(
                    (__m256i *)(in2_mat+i_dot*n_col8_2+i_col));
                acc = _mm256_xor_si256(
                    acc, tbm_mult8x8_m256i_gfnio(repeat, b_8x8_4));
            }
            _mm256_maskstore_epi64(
                (long long int *)(out_mat+i_row*n_col8_2+i_col), mask, acc);
        }
    }
}

#pragma GCC push_options //-----------------------------------------------------
#pragma GCC optimize("no-tree-vectorize")
void __attribute__ ((noinline)) tbm_mult_gfnio_ncol8_1(
    uint64_t *in, uint64_t n_mat, uint64_t *in2, uint64_t *out)
{
#if defined(USE_DOT)
    tbm_mult_t_gfnio_dot(in, n_mat, 1, 1, in2, 1, out);
#else
    uint64_t i8x8; //!< index for 4 8x8 blocks
    for (i8x8 = 0; i8x8 < n_mat/4*4; i8x8 += 4)
    {
        __m256i in8x8_4 = _mm256_loadu_si256((__m256i *)(in+i8x8));
        __m256i in2_8x8_4 = _mm256_loadu_si256((__m256i *)(in2+i8x8));
        _mm256_storeu_si256(
            (__m256i *)(out+i8x8),
            tbm_mult8x8_m256i_gfnio(in8x8_4, in2_8x8_4));
    }

    if (i8x8 == n_mat)
        return; // all blocks are processed
    __m256i in8x8_4 = _mm256_loadu_si256((__m256i *)(in+i8x8));
    __m256i in2_8x8_4 = _mm256_loadu_si256((__m256i *)(in2+i8x8));
    __m256i mask = _mm256_set_epi64x(3, 2, 1, 0); //!< mask for the last block
    mask = _mm256_cmpgt_epi64(_mm256_set1_epi64x(n_mat-i8x8), mask);
    _mm256_maskstore_epi64(
        (long long int *)(out+i8x8), mask, 
        tbm_mult8x8_m256i_gfnio(in8x8_4, in2_8x8_4));
#endif
}

void __attribute__ ((noinline)) tbm_mult_gfnio_ncol8_2(
    uint64_t *in, uint64_t n_mat, uint64_t *in2, uint64_t *out)
{
#if defined(USE_DOT)
    tbm_mult_t_gfnio_dot(in, n_mat, 2, 2, in2, 2, out);
#else
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        __m256i in16X16 = _mm256_loadu_si256((__m256i *)in);
        __m256i in2_16X16 = _mm256_loadu_si256((__m256i *)in2);
        _mm256_storeu_si256(
            (__m256i *)out, tbm_mult16x16_m256i_gfnio(in16X16, in2_16X16));
        in += 4;
        in2 += 4;
        out += 4;
    }
#endif
}

void __attribute__ ((noinline)) tbm_mult_gfnio_ncol8_4(
    uint64_t *in, uint64_t n_mat, uint64_t *in2, uint64_t *out)
{
#if defined(USE_DOT)
    tbm_mult_t_gfnio_dot(in, n_mat, 4, 4, in2, 4, out);
#else
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        __m256i in8x32[4];
        __m256i in2_8x32[4];
        for (uint8_t i_8row = 0; i_8row < 4; i_8row++)
        {
            in8x32[i_8row] = _mm256_loadu_si256(((__m256i *)in)+i_8row);
            in2_8x32[i_8row] = _mm256_loadu_si256(((__m256i *)in2)+i_8row);
        }
        tbm_mult32x32_m256i_gfnio(in8x32, in2_8x32);
        for (uint8_t i_8row = 0; i_8row < 4; i_8row++)
            _mm256_storeu_si256(((__m256i *)out)+i_8row, in8x32[i_8row]);
        in += 16;
        in2 += 16;
        out += 16;
    }
#endif
}

void tbm_mult_gfnio(
    uint64_t *in, uint64_t n_mat, uint32_t n_row8, uint32_t n_col8,
    uint64_t *in2, uint32_t n_col8_2, uint64_t *out)
{
    if ((n_col8 == 1) && (n_row8 == 1) && (n_col8_2 == 1))
        return tbm_mult_gfnio_ncol8_1(in, n_mat, in2, out);
    // if ((n_col8 == 2) && (n_row8 == 2) && (n_col8_2 == 2))
    //     return tbm_mult_t_gfnio_ncol8_2(in, n_mat, in2, out);
    // if ((n_col8 == 4) && (n_row8 == 4) && (n_col8_2 == 4))
    //     return tbm_mult_t_gfnio_ncol8_4(in, n_mat, in2, out);
    
    tbm_mult_gfnio_256(in, n_mat, n_row8, n_col8, in2, n_col8_2, out);
}
#pragma GCC pop_options //------------------------------------------------------

//______________________________________________________________________________
__m256i inline tbm_mult_t8x8_m256i_gfnio(__m256i a, __m256i b)
{
    // _mm256_gf2p8affine_epi64_epi8(B, A, 0) is (A*B.T).T
    // _mm256_gf2p8affine_epi64_epi8(A, B, 0) is (B*A.T).T = A*B.T
    return _mm256_gf2p8affine_epi64_epi8(a, b, 0);
}

uint64_t inline tbm_dot_t_gfnio(uint64_t *in, uint64_t *in2, uint32_t n_col8)
{
    uint64_t i8x8; //!< index for 4 8x8 blocks
    uint64_t n8x8 = n_col8; //!< number of 8x8 blocks
    __m256i in8x8_4; //!< 4 8x8 blocks from the first matrix
    __m256i in2_8x8_4; //!< 4 8x8 blocks from the second transposed matrix
__m256i acc = _mm256_setzero_si256(); //!< accumulator for 4 8x8 blocks
    for (i8x8 = 0; i8x8 < n8x8/4*4; i8x8 += 4)
    {
        // load 4x8x8 blocks
        in8x8_4 = _mm256_loadu_si256((__m256i *)(in+i8x8));
        in2_8x8_4 = _mm256_loadu_si256((__m256i *)(in2+i8x8));
        // transpose 4x8x8 blocks
        __m256i out8x8_4 = tbm_mult_t8x8_m256i_gfnio(in8x8_4, in2_8x8_4);
        // store transposed 4x8x8 blocks
        acc = _mm256_xor_si256(acc, out8x8_4);
    }
    if (i8x8 != n8x8)
    {
        __m256i mask; //!< mask for the last block
        mask = _mm256_set_epi64x(3, 2, 1, 0);
        mask = _mm256_cmpgt_epi64(_mm256_set1_epi64x(n8x8-i8x8), mask);
        in8x8_4 = _mm256_maskload_epi64((long long int *)(in+i8x8), mask);
        in2_8x8_4 = _mm256_maskload_epi64((long long int *)(in2+i8x8), mask);
        __m256i out8x8_4 = tbm_mult_t8x8_m256i_gfnio(in8x8_4, in2_8x8_4);
        acc = _mm256_xor_si256(acc, out8x8_4);
    }
    __m128i acc128 = _mm256_extracti128_si256(acc, 0);
    acc128 = _mm_xor_si128(acc128, _mm256_extracti128_si256(acc, 1));
    return _mm_extract_epi64(acc128, 0) ^ _mm_extract_epi64(acc128, 1);
}

__m256i inline tbm_mult_t16x16_m256i_gfnio(__m256i a3210, __m256i b3210)
{
    // a is:    b.T is:
    // [0, 1]   [0, 2]   [0, 0]   [0, 2]   [1, 1]   [1, 3]
    // [2, 3] @ [1, 3] = [2, 2] * [0, 2] + [3, 3] * [1, 3]
    __m256i a3311 = _mm256_permute4x64_epi64(a3210, _MM_SHUFFLE(3, 3, 1, 1));
    __m256i a2200 = _mm256_permute4x64_epi64(a3210, _MM_SHUFFLE(2, 2, 0, 0));
    __m256i b3131 = _mm256_permute4x64_epi64(b3210, _MM_SHUFFLE(3, 1, 3, 1));
    __m256i b2020 = _mm256_permute4x64_epi64(b3210, _MM_SHUFFLE(2, 0, 2, 0));

    __m256i out = _mm256_xor_si256(
        _mm256_gf2p8affine_epi64_epi8(a3311, b3131, 0),
        _mm256_gf2p8affine_epi64_epi8(a2200, b2020, 0));

    return out;
}

void inline tbm_mult_t32x32_m256i_gfnio(__m256i a[4], __m256i b[4])
{
    __m256i b_3210 = b[0];
    __m256i b_7654 = b[1];
    __m256i b_ba98 = b[2];
    __m256i b_fedc = b[3];

    __m256i b_9810 = _mm256_permute2x128_si256(b_3210, b_ba98, 0x20);
    __m256i b_dc54 = _mm256_permute2x128_si256(b_7654, b_fedc, 0x20);
    __m256i b_ba32 = _mm256_permute2x128_si256(b_3210, b_ba98, 0x31);
    __m256i b_fe76 = _mm256_permute2x128_si256(b_7654, b_fedc, 0x31);

    __m256i b_c840 = _mm256_unpacklo_epi64(b_9810, b_dc54);
    __m256i b_d951 = _mm256_unpackhi_epi64(b_9810, b_dc54);
    __m256i b_ea62 = _mm256_unpacklo_epi64(b_ba32, b_fe76);
    __m256i b_fb73 = _mm256_unpackhi_epi64(b_ba32, b_fe76);

    uint64_t *a64 = (uint64_t *)a;
    for (uint8_t i_row = 0; i_row < 4; i_row++)
    {
        // __m256i a3210 = a[i_row];

        __m256i repeat; //<! current product of a cell 4 times  
        __m256i prod; //<! current product of a cell and b row
        __m256i out; //<! accumulated sum of the products
        out = _mm256_setzero_si256();
        // repeat = _mm256_permute4x64_epi64(a3210, _MM_SHUFFLE(0, 0, 0, 0));
        repeat = _mm256_set1_epi64x(*a64++);
        prod = _mm256_gf2p8affine_epi64_epi8(repeat, b_c840, 0);
        out = _mm256_xor_si256(out, prod);

        // repeat = _mm256_permute4x64_epi64(a3210, _MM_SHUFFLE(1, 1, 1, 1));
        repeat = _mm256_set1_epi64x(*a64++);
        prod = _mm256_gf2p8affine_epi64_epi8(repeat, b_d951, 0);
        out = _mm256_xor_si256(out, prod);

        // repeat = _mm256_permute4x64_epi64(a3210, _MM_SHUFFLE(2, 2, 2, 2));
        repeat = _mm256_set1_epi64x(*a64++);
        prod = _mm256_gf2p8affine_epi64_epi8(repeat, b_ea62, 0);
        out = _mm256_xor_si256(out, prod);

        // repeat = _mm256_permute4x64_epi64(a3210, _MM_SHUFFLE(3, 3, 3, 3));
        repeat = _mm256_set1_epi64x(*a64++);
        prod = _mm256_gf2p8affine_epi64_epi8(repeat, b_fb73, 0);
        out = _mm256_xor_si256(out, prod);

        a[i_row] = out;
    }
}

void inline tbm_mult_t_gfnio_dot(
    uint64_t *in, uint64_t n_mat, uint32_t n_row8, uint32_t n_col8,
    uint64_t *in2, uint32_t n_row8_2, uint64_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        uint64_t *in_mat = in + i_mat*n_row8*n_col8;
        uint64_t *in2_mat = in2 + i_mat*n_row8_2*n_col8;
        uint64_t *out_mat = out + i_mat*n_row8*n_row8_2;
        for (uint32_t i_row = 0; i_row < n_row8; i_row++)
            for (uint32_t i_row2 = 0; i_row2 < n_row8_2; i_row2++)
                out_mat[i_row*n_row8_2+i_row2] = tbm_dot_t_gfnio(
                    in_mat+i_row*n_col8, in2_mat+i_row2*n_col8, n_col8);
    }
}

#pragma GCC push_options //-----------------------------------------------------
#pragma GCC optimize("no-tree-vectorize")
void __attribute__ ((noinline)) tbm_mult_t_gfnio_ncol8_1(
    uint64_t *in, uint64_t n_mat, uint64_t *in2, uint64_t *out)
{
#if defined(USE_DOT)
    tbm_mult_t_gfnio_dot(in, n_mat, 1, 1, in2, 1, out);
#else
    uint64_t i8x8; //!< index for 4 8x8 blocks
    for (i8x8 = 0; i8x8 < n_mat/4*4; i8x8 += 4)
    {
        __m256i in8x8_4 = _mm256_loadu_si256((__m256i *)(in+i8x8));
        __m256i in2_8x8_4 = _mm256_loadu_si256((__m256i *)(in2+i8x8));
        _mm256_storeu_si256(
            (__m256i *)(out+i8x8),
            tbm_mult_t8x8_m256i_gfnio(in8x8_4, in2_8x8_4));
    }

    if (i8x8 == n_mat)
        return; // all blocks are processed
    __m256i in8x8_4 = _mm256_loadu_si256((__m256i *)(in+i8x8));
    __m256i in2_8x8_4 = _mm256_loadu_si256((__m256i *)(in2+i8x8));
    __m256i mask = _mm256_set_epi64x(3, 2, 1, 0); //!< mask for the last block
    mask = _mm256_cmpgt_epi64(_mm256_set1_epi64x(n_mat-i8x8), mask);
    _mm256_maskstore_epi64(
        (long long int *)(out+i8x8), mask, 
        tbm_mult_t8x8_m256i_gfnio(in8x8_4, in2_8x8_4));
#endif
}

void __attribute__ ((noinline)) tbm_mult_t_gfnio_ncol8_2(
    uint64_t *in, uint64_t n_mat, uint64_t *in2, uint64_t *out)
{
#if defined(USE_DOT)
    tbm_mult_t_gfnio_dot(in, n_mat, 2, 2, in2, 2, out);
#else
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        __m256i in16X16 = _mm256_loadu_si256((__m256i *)in);
        __m256i in2_16X16 = _mm256_loadu_si256((__m256i *)in2);
        _mm256_storeu_si256(
            (__m256i *)out, tbm_mult_t16x16_m256i_gfnio(in16X16, in2_16X16));
        in += 4;
        in2 += 4;
        out += 4;
    }
#endif
}

void __attribute__ ((noinline)) tbm_mult_t_gfnio_ncol8_4(
    uint64_t *in, uint64_t n_mat, uint64_t *in2, uint64_t *out)
{
#if defined(USE_DOT)
    tbm_mult_t_gfnio_dot(in, n_mat, 4, 4, in2, 4, out);
#else
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        __m256i in8x32[4];
        __m256i in2_8x32[4];
        for (uint8_t i_8row = 0; i_8row < 4; i_8row++)
        {
            in8x32[i_8row] = _mm256_loadu_si256(((__m256i *)in)+i_8row);
            in2_8x32[i_8row] = _mm256_loadu_si256(((__m256i *)in2)+i_8row);
        }
        tbm_mult_t32x32_m256i_gfnio(in8x32, in2_8x32);
        for (uint8_t i_8row = 0; i_8row < 4; i_8row++)
            _mm256_storeu_si256(((__m256i *)out)+i_8row, in8x32[i_8row]);
        in += 16;
        in2 += 16;
        out += 16;
    }
#endif
}

void tbm_mult_t_gfnio(
    uint64_t *in, uint64_t n_mat, uint32_t n_row8, uint32_t n_col8,
    uint64_t *in2, uint32_t n_row8_2, uint64_t *out)
{
    if ((n_col8 == 1) && (n_row8 == 1) && (n_row8_2 == 1))
        return tbm_mult_t_gfnio_ncol8_1(in, n_mat, in2, out);
    if ((n_col8 == 2) && (n_row8 == 2) && (n_row8_2 == 2))
        return tbm_mult_t_gfnio_ncol8_2(in, n_mat, in2, out);
    if ((n_col8 == 4) && (n_row8 == 4) && (n_row8_2 == 4))
        return tbm_mult_t_gfnio_ncol8_4(in, n_mat, in2, out);
    
    tbm_mult_t_gfnio_dot(in, n_mat, n_row8, n_col8, in2, n_row8_2, out);
}
#pragma GCC pop_options //------------------------------------------------------
