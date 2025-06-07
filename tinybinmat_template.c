typedef void tbm_transpose8x8_x4_fun_t(uint64_t in8x8[4]);
typedef void tbm_mult8x8_1x4_fun_t(uint64_t a, uint64_t b[4], uint64_t out[4]);
typedef void tbm_mult8x8_x4_fun_t(
    uint64_t a[4], uint64_t b[4], uint64_t out[4]);
typedef uint64_t tbm_dot_fun_t(uint64_t *in, uint64_t *in2, uint32_t n_col8);

static void inline tbm_transpose_256_template(
    uint64_t *in, uint64_t n_mat, uint32_t n_row8, uint32_t n_col8, 
    uint64_t *out, tbm_transpose8x8_x4_fun_t *transpose_x4_fun)
{
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
        transpose_x4_fun(out+i8x8);
    if (i8x8 == n8x8)
        return; // all blocks are processed
    uint64_t tmp[4];
    for (uint8_t i_prod = 0; i_prod < (n8x8 & 3); i_prod++)
        tmp[i_prod] = out[i8x8+i_prod];
    transpose_x4_fun(tmp);
    for (uint8_t i_prod = 0; i_prod < (n8x8 & 3); i_prod++)
        out[i8x8+i_prod] = tmp[i_prod];
}

static void inline tbm_mult32x32_template(
    uint64_t a[16], uint64_t b[16], uint64_t out[16], 
    tbm_mult8x8_1x4_fun_t *mult1x4_fun)
{
    uint64_t *a64 = a;
    for (uint8_t i_row = 0; i_row < 4; i_row++)
    {
        uint64_t acc[4];
        uint64_t prod[4]; //!< current product of a cell and b row
        mult1x4_fun(*a64++, b+0, acc);
        mult1x4_fun(*a64++, b+4, prod);
        for (uint8_t i_prod = 0; i_prod < 4; i_prod++)
            acc[i_prod] ^= prod[i_prod];
        mult1x4_fun(*a64++, b+8, prod);
        for (uint8_t i_prod = 0; i_prod < 4; i_prod++)
            acc[i_prod] ^= prod[i_prod];
        mult1x4_fun(*a64++, b+12, prod);
        for (uint8_t i_prod = 0; i_prod < 4; i_prod++)
            out[4*i_row+i_prod] = acc[i_prod] ^ prod[i_prod];
    }
}

static void inline tbm_mult_256_template(
    uint64_t *in, uint64_t n_mat, uint32_t n_row8, uint32_t n_col8,
    uint64_t *in2, uint32_t n_col8_2, uint64_t *out, 
    tbm_mult8x8_1x4_fun_t *mult1x4_fun)
{
    __m256i mask = _mm256_set_epi64x(3, 2, 1, 0); //!< mask for the last block
    mask = _mm256_cmpgt_epi64(_mm256_set1_epi64x(n_col8_2-n_col8_2/4*4), mask);
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        uint64_t *in_mat = in + i_mat*n_row8*n_col8;
        uint64_t *in2_mat = in2 + i_mat*n_col8*n_col8_2;
        uint64_t *out_mat = out + i_mat*n_row8*n_col8_2;
        uint32_t i_row; //!< index of output row
        for (i_row = 0; i_row < n_row8; i_row++)
        {
            uint32_t i_col; //!< index of output colum
            for (i_col = 0; i_col < n_col8_2/4*4; i_col += 4)
            {
                uint64_t acc[4] = {0, 0, 0, 0};
                for (uint32_t i_dot = 0; i_dot < n_col8; i_dot++)
                {
                    uint64_t prod[4];
                    mult1x4_fun(
                        in_mat[i_row*n_col8+i_dot], 
                        in2_mat+i_dot*n_col8_2+i_col, prod);
                    for (uint8_t i_prod = 0; i_prod < 4; i_prod++)
                        acc[i_prod] ^= prod[i_prod];
                }
                for (uint8_t i_prod = 0; i_prod < 4; i_prod++)
                    out_mat[i_row*n_col8_2+i_col+i_prod] = acc[i_prod];
            }
            if (i_col == n_col8_2)
                continue; // all blocks are processed
            uint64_t acc[4] = {0, 0, 0, 0};
            for (uint32_t i_dot = 0; i_dot < n_col8; i_dot++)
            {
                uint64_t prod[4];
                mult1x4_fun(
                    in_mat[i_row*n_col8+i_dot], 
                    in2_mat+i_dot*n_col8_2+i_col, prod);
                for (uint8_t i_prod = 0; i_prod < 4; i_prod++)
                    acc[i_prod] ^= prod[i_prod];
            }
            for (uint8_t i_prod = 0; i_prod < (n_col8_2 & 3); i_prod++)
                out_mat[i_row*n_col8_2+i_col+i_prod] = acc[i_prod];
        }
    }
}

void inline tbm_mult_t_dot_template(
    uint64_t *in, uint64_t n_mat, uint32_t n_row8, uint32_t n_col8,
    uint64_t *in2, uint32_t n_row8_2, uint64_t *out, tbm_dot_fun_t *dot_t_fun)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        uint64_t *in_mat = in + i_mat*n_row8*n_col8;
        uint64_t *in2_mat = in2 + i_mat*n_row8_2*n_col8;
        uint64_t *out_mat = out + i_mat*n_row8*n_row8_2;
        for (uint32_t i_row = 0; i_row < n_row8; i_row++)
            for (uint32_t i_row2 = 0; i_row2 < n_row8_2; i_row2++)
                out_mat[i_row*n_row8_2+i_row2] = dot_t_fun(
                    in_mat+i_row*n_col8, in2_mat+i_row2*n_col8, n_col8);
    }
}

