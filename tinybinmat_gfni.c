#include "tinybinmat.h"

//______________________________________________________________________________
__m256i inline tbm_transpose8x8_m256i_gfni(__m256i in8x8_4)
{
    // _mm256_gf2p8affine_epi64_epi8(I, A, 0) is (A*I.T).T = A.T
    // flipud of the matrix are needed before and after the transformation
    // as conventions are different: J*gf2p8affine(I, J*A, 0) = J*(J*A).T
    // _mm256_gf2p8affine_epi64_epi8(J, J*A, 0) is ((J*A)*J.T).T = J*(J*A).T
    // saves a flipud
    __m128i reverse8_2col = _mm_set_epi8(
        8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7);
    __m256i reverse8_col = _mm256_set_m128i(reverse8_2col, reverse8_2col);
    __m256i in8x8_4rev = _mm256_shuffle_epi8(in8x8_4, reverse8_col);

    __m256i neye_8x8_4 = _mm256_set1_epi64x(0x8040201008040201);
    return _mm256_gf2p8affine_epi64_epi8(neye_8x8_4, in8x8_4rev, 0);
}

__m256i inline tbm_transpose16x16_m256i_gfni(__m256i a)
{
    __m128i reverse8_2col = _mm_set_epi8(
        8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7);
    __m256i reverse8_col = _mm256_set_m128i(reverse8_2col, reverse8_2col);

    // We want to convert the 16x16 matrix to four 8x8 matrix
    // following the order: [[sub1, sub0], [sub3, sub2]]
    // first 16 bits are in least signicant bir order: b15 down to b0
    // odd and even octets represent submatrices sub1 and sub0
    // but as 16x16 matrix encodes a row as [b0, b1, ... b15]
    // matrix sub0 is actually located in the odd octets
    __m128i split8x8_2 = _mm_set_epi8(
        14, 12, 10, 8, 6, 4, 2, 0, 15, 13, 11, 9, 7, 5, 3, 1);
    __m256i split8x8_4 = _mm256_set_m128i(split8x8_2, split8x8_2);
    __m256i split8x8_4r = _mm256_shuffle_epi8(split8x8_4, reverse8_col);
    __m256i a3210r = _mm256_shuffle_epi8(a, split8x8_4r);

    __m256i a0213r = _mm256_permute4x64_epi64(a3210r, _MM_SHUFFLE(0, 2, 1, 3));
    __m256i neye_8x8_4 = _mm256_set1_epi64x(0x8040201008040201);
    __m256i out = _mm256_gf2p8affine_epi64_epi8(neye_8x8_4, a0213r, 0);

    __m128i unsplit8x8_2 = _mm_set_epi8(
        7, 15, 6, 14, 5, 13, 4, 12, 3, 11, 2, 10, 1, 9, 0, 8);
    __m256i unsplit8x8_4 = _mm256_set_m128i(unsplit8x8_2, unsplit8x8_2);
    return _mm256_shuffle_epi8(out, unsplit8x8_4);
}

void inline tbm_transpose32x32_m256i_gfni(__m256i in_read[4], __m256i in[4])
{
    __m128i split8x8_2r = _mm_set_epi8(
        0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    __m256i split8x8_4r = _mm256_set_m128i(split8x8_2r, split8x8_2r);
    __m256i split8x8_4r_128 = _mm256_set_epi32(3, 7, 2, 6, 1, 5, 0, 4);

    __m128i unsplit8x8_2 = _mm_set_epi8(
        3, 7, 11, 15, 2, 6, 10, 14, 1, 5, 9, 13, 0, 4, 8, 12);
    __m256i unsplit8x8_4 = _mm256_set_m128i(unsplit8x8_2, unsplit8x8_2);
    __m256i unsplit8x8_4_128 = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);

    __m256i b3210r = _mm256_shuffle_epi8(in_read[0], split8x8_4r);
    __m256i b7654r = _mm256_shuffle_epi8(in_read[1], split8x8_4r);
    __m256i bba98r = _mm256_shuffle_epi8(in_read[2], split8x8_4r);
    __m256i bfedcr = _mm256_shuffle_epi8(in_read[3], split8x8_4r);
    b3210r = _mm256_permutevar8x32_epi32(b3210r, split8x8_4r_128);
    b7654r = _mm256_permutevar8x32_epi32(b7654r, split8x8_4r_128);
    bba98r = _mm256_permutevar8x32_epi32(bba98r, split8x8_4r_128);
    bfedcr = _mm256_permutevar8x32_epi32(bfedcr, split8x8_4r_128);

    __m256i b32bar = _mm256_permute2x128_si256(b3210r, bba98r, 0x13);
    __m256i b76fer = _mm256_permute2x128_si256(b7654r, bfedcr, 0x13);
    __m256i b1098r = _mm256_permute2x128_si256(b3210r, bba98r, 0x02);
    __m256i b54dcr = _mm256_permute2x128_si256(b7654r, bfedcr, 0x02);

    __m256i b37bfr = _mm256_unpackhi_epi64 (b76fer, b32bar);
    __m256i b26aer = _mm256_unpacklo_epi64 (b76fer, b32bar);
    __m256i b159dr = _mm256_unpackhi_epi64 (b54dcr, b1098r);
    __m256i b048cr = _mm256_unpacklo_epi64 (b54dcr, b1098r);

    __m256i neye_8x8_4 = _mm256_set1_epi64x(0x8040201008040201);
    __m256i out = _mm256_gf2p8affine_epi64_epi8(neye_8x8_4, b37bfr, 0);
    out = _mm256_permutevar8x32_epi32(out, unsplit8x8_4_128);
    out = _mm256_shuffle_epi8(out, unsplit8x8_4);
    in[0] = out;

    out = _mm256_gf2p8affine_epi64_epi8(neye_8x8_4, b26aer, 0);
    out = _mm256_permutevar8x32_epi32(out, unsplit8x8_4_128);
    out = _mm256_shuffle_epi8(out, unsplit8x8_4);
    in[1] = out;

    out = _mm256_gf2p8affine_epi64_epi8(neye_8x8_4, b159dr, 0);
    out = _mm256_permutevar8x32_epi32(out, unsplit8x8_4_128);
    out = _mm256_shuffle_epi8(out, unsplit8x8_4);
    in[2] = out;

    out = _mm256_gf2p8affine_epi64_epi8(neye_8x8_4, b048cr, 0);
    out = _mm256_permutevar8x32_epi32(out, unsplit8x8_4_128);
    out = _mm256_shuffle_epi8(out, unsplit8x8_4);
    in[3] = out;
}

#pragma GCC push_options //----------------------------------------------------
#pragma GCC optimize("no-tree-vectorize")
void tbm_transpose8x8_gfni(uint8_t *in, uint64_t n_mat, uint8_t *out)
{
    uint64_t i_avx2 = n_mat/4*4;
    for (uint64_t i_mat = 0; i_mat < i_avx2; i_mat += 4)
    {
        __m256i in8x8_4 = _mm256_loadu_si256((__m256i *)in);
        __m256i out8x8_4 = tbm_transpose8x8_m256i_gfni(in8x8_4);
        _mm256_storeu_si256((__m256i *)out, out8x8_4);
        in += 8*4;
        out += 8*4;
    }
    for (uint64_t i_mat = i_avx2; i_mat < n_mat; i_mat++)
    {
        *((uint64_t *)out) = tbm_transpose8x8_uint64(*((uint64_t *)in));
        in += 8;
        out += 8;
    }
}

void tbm_transpose16x16_gfni(uint16_t *in, uint64_t n_mat, uint16_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        __m256i in16x16 = _mm256_loadu_si256((__m256i *)in);
        __m256i out16x16 = tbm_transpose16x16_m256i_gfni(in16x16);
        _mm256_storeu_si256((__m256i *)out, out16x16);
        in += 16;
        out += 16;
    }
}

void tbm_transpose32x32_gfni(uint32_t *in, uint64_t n_mat, uint32_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        __m256i in8x32[4];
        for (uint8_t i_8row = 0; i_8row < 4; i_8row++)
            in8x32[i_8row] = _mm256_loadu_si256(((__m256i *)in)+i_8row);
        tbm_transpose32x32_m256i_gfni(in8x32, in8x32);
        for (uint8_t i_8row = 0; i_8row < 4; i_8row++)
            _mm256_storeu_si256(((__m256i *)out)+i_8row, in8x32[i_8row]);
        in += 32;
        out += 32;
    }
}
#pragma GCC pop_options //-----------------------------------------------------

//______________________________________________________________________________
// multiply 4 groups of two 8x8 bit matrices
__m256i inline tbm_mult8x8_m256i_gfni(__m256i a, __m256i b)
{
    // _mm256_gf2p8affine_epi64_epi8(B, A, 0) is (A*B.T).T
    // _mm256_gf2p8affine_epi64_epi8(A, B.T, 0) is (B.T*A.T).T = A*B
    // the second form needs a single transposition
    // J*_mm256_gf2p8affine_epi64_epi8(J*A, (J*B).T, 0) is A*(J*B)
    __m128i reverse8_2col = _mm_set_epi8(
        8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7);
    __m256i reverse8_col = _mm256_set_m128i(reverse8_2col, reverse8_2col);
    __m256i b8x8_4rev = _mm256_shuffle_epi8(b, reverse8_col);

    __m256i eye_8x8_4 = _mm256_set1_epi64x(0x0102040810204080);
    __m256i b8x8_4t = _mm256_gf2p8affine_epi64_epi8(eye_8x8_4, b8x8_4rev, 0);
    return _mm256_gf2p8affine_epi64_epi8(a, b8x8_4t, 0);
}

__m256i inline tbm_mult16x16_m256i_gfni(__m256i a, __m256i b)
{
    __m128i reverse8_2col = _mm_set_epi8(
        8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7);
    __m256i reverse8_col = _mm256_set_m128i(reverse8_2col, reverse8_2col);
    __m256i eye_8x8_4 = _mm256_set1_epi64x(0x0102040810204080);

    // We want to convert the 16x16 matrix to four 8x8 matrix
    // following the order: [[sub1, sub0], [sub3, sub2]]
    // first 16 bits are in least signicant bir order: b15 down to b0
    // odd and even octets represent submatrices sub1 and sub0
    // but as 16x16 matrix encodes a row as [b0, b1, ... b15]
    // matrix sub0 is actually located in the odd octets
    __m128i split8x8_2 = _mm_set_epi8(
        14, 12, 10, 8, 6, 4, 2, 0, 15, 13, 11, 9, 7, 5, 3, 1);
    __m256i split8x8_4 = _mm256_set_m128i(split8x8_2, split8x8_2);
    __m256i split8x8_4r = _mm256_shuffle_epi8(split8x8_4, reverse8_col);
    __m256i a3210 = _mm256_shuffle_epi8(a, split8x8_4);
    __m256i b3210r = _mm256_shuffle_epi8(b, split8x8_4r);
    __m256i b3210t = _mm256_gf2p8affine_epi64_epi8(eye_8x8_4, b3210r, 0);

    __m256i a3311 = _mm256_permute4x64_epi64(a3210, _MM_SHUFFLE(3, 3, 1, 1));
    __m256i a2200 = _mm256_permute4x64_epi64(a3210, _MM_SHUFFLE(2, 2, 0, 0));
    __m256i b1010t = _mm256_permute4x64_epi64(b3210t, _MM_SHUFFLE(1, 0, 1, 0));
    __m256i b3232t = _mm256_permute4x64_epi64(b3210t, _MM_SHUFFLE(3, 2, 3, 2));

    // __m256i out = _mm256_xor_si256(
    //     tbm_mult8x8_m256i_gfni(a3311, b1010),
    //     tbm_mult8x8_m256i_gfni(a2200, b3232));

    __m256i out = _mm256_xor_si256(
        _mm256_gf2p8affine_epi64_epi8(a3311, b1010t, 0),
        _mm256_gf2p8affine_epi64_epi8(a2200, b3232t, 0));

    __m128i unsplit8x8_2 = _mm_set_epi8(
        7, 15, 6, 14, 5, 13, 4, 12, 3, 11, 2, 10, 1, 9, 0, 8);
    __m256i unsplit8x8_4 = _mm256_set_m128i(unsplit8x8_2, unsplit8x8_2);
    return _mm256_shuffle_epi8(out, unsplit8x8_4);
}

void inline tbm_mult32x32_m256i_gfni(__m256i a[4], __m256i b[4], __m256i out[4])
{
    // We want to convert the 8x32 matrix to four 8x8 matrix
    // following the order: [[sub1, sub0], [sub3, sub2]]
    // first 32 bits are in least signicant bir order: b31 down to b0
    // modulo-4 octets represent submatrices sub3 to sub0
    // but as 32x32 matrix encodes a row as [b0, b1, ... b31]
    // matrix sub0 is actually located in 3 modulo 4 octets
    __m128i split8x8_2 = _mm_set_epi8(
        12, 8, 4, 0, 13, 9, 5, 1, 14, 10, 6, 2, 15, 11, 7, 3);
    __m256i split8x8_4 = _mm256_set_m128i(split8x8_2, split8x8_2);
    __m256i split8x8_4_128 = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);

    // convert format for matrix a
    for (uint8_t i_row = 0; i_row < 4; i_row++)
    {
        __m256i a3210 = _mm256_shuffle_epi8(a[i_row], split8x8_4);
        a[i_row] = _mm256_permutevar8x32_epi32(a3210, split8x8_4_128);
    }
    
    __m128i split8x8_2r = _mm_set_epi8(
        0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    __m256i split8x8_4r = _mm256_set_m128i(split8x8_2r, split8x8_2r);
    __m256i split8x8_4r_128 = _mm256_set_epi32(3, 7, 2, 6, 1, 5, 0, 4);

    __m128i unsplit8x8_2 = _mm_set_epi8(
        3, 7, 11, 15, 2, 6, 10, 14, 1, 5, 9, 13, 0, 4, 8, 12);
    __m256i unsplit8x8_4 = _mm256_set_m128i(unsplit8x8_2, unsplit8x8_2);
    __m256i unsplit8x8_4_128 = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);

    __m256i b3210r = _mm256_shuffle_epi8(b[0], split8x8_4r);
    __m256i b7654r = _mm256_shuffle_epi8(b[1], split8x8_4r);
    __m256i bba98r = _mm256_shuffle_epi8(b[2], split8x8_4r);
    __m256i bfedcr = _mm256_shuffle_epi8(b[3], split8x8_4r);
    b3210r = _mm256_permutevar8x32_epi32(b3210r, split8x8_4r_128);
    b7654r = _mm256_permutevar8x32_epi32(b7654r, split8x8_4r_128);
    bba98r = _mm256_permutevar8x32_epi32(bba98r, split8x8_4r_128);
    bfedcr = _mm256_permutevar8x32_epi32(bfedcr, split8x8_4r_128);

    __m256i eye_8x8_4 = _mm256_set1_epi64x(0x0102040810204080);
    __m256i b3210t = _mm256_gf2p8affine_epi64_epi8(eye_8x8_4, b3210r, 0);
    __m256i b7654t = _mm256_gf2p8affine_epi64_epi8(eye_8x8_4, b7654r, 0);
    __m256i bba98t = _mm256_gf2p8affine_epi64_epi8(eye_8x8_4, bba98r, 0);
    __m256i bfedct = _mm256_gf2p8affine_epi64_epi8(eye_8x8_4, bfedcr, 0);

    uint64_t *a64 = (uint64_t *)a;
    for (uint8_t i_row = 0; i_row < 4; i_row++)
    {
        __m256i repeat; //<! current product of a cell 4 times  
        __m256i prod; //<! current product of a cell and b row
        __m256i acc; //<! accumulated sum of the products
        acc = _mm256_setzero_si256();
        repeat = _mm256_set1_epi64x(*a64++);
        prod = _mm256_gf2p8affine_epi64_epi8(repeat, bfedct, 0);
        acc = _mm256_xor_si256(acc, prod);

        repeat = _mm256_set1_epi64x(*a64++);
        prod = _mm256_gf2p8affine_epi64_epi8(repeat, bba98t, 0);
        acc = _mm256_xor_si256(acc, prod);

        repeat = _mm256_set1_epi64x(*a64++);
        prod = _mm256_gf2p8affine_epi64_epi8(repeat, b7654t, 0);
        acc = _mm256_xor_si256(acc, prod);

        repeat = _mm256_set1_epi64x(*a64++);
        prod = _mm256_gf2p8affine_epi64_epi8(repeat, b3210t, 0);
        acc = _mm256_xor_si256(acc, prod);

        acc = _mm256_permutevar8x32_epi32(acc, unsplit8x8_4_128);
        acc = _mm256_shuffle_epi8(acc, unsplit8x8_4);
        out[i_row] = acc;
    }
}

#pragma GCC push_options //----------------------------------------------------
#pragma GCC optimize("no-tree-vectorize")
void tbm_mult8x8_gfni(uint8_t *in, uint8_t *in2, uint64_t n_mat, uint8_t *out)
{
    uint64_t i_avx2 = 0;
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
    for (uint64_t i_mat = i_avx2; i_mat < n_mat; i_mat++)
    {
        *((uint64_t *)out) = tbm_mult8x8_uint64(*((uint64_t *)in), in2);
        in += 8;
        in2 += 8;
        out += 8;
    }
}

void tbm_mult16x16_gfni(
    uint16_t *in, uint16_t *in2, uint64_t n_mat, uint16_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        __m256i in16x16 = _mm256_loadu_si256((__m256i *)in);
        __m256i in2_16x16 = _mm256_loadu_si256((__m256i *)in2);
        __m256i out16x16 = tbm_mult16x16_m256i_gfni(in16x16, in2_16x16);
        // __m256i out16x16 = tbm_mult16x16_m256i(in16x16, in2);
        _mm256_storeu_si256((__m256i *)out, out16x16);
        in += 16;
        in2 += 16;
        out += 16;
    }
}

void tbm_mult32x32_gfni(
    uint32_t *in, uint32_t *in2, uint64_t n_mat, uint32_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        __m256i in8x32[4];
        __m256i in2_8x32[4];
        __m256i out8x32[4];
        for (uint8_t i_8row = 0; i_8row < 4; i_8row++)
        {
            in8x32[i_8row] = _mm256_loadu_si256(((__m256i *)in)+i_8row);
            in2_8x32[i_8row] = _mm256_loadu_si256(((__m256i *)in2)+i_8row);
        }
        tbm_mult32x32_m256i_gfni(in8x32, in2_8x32, out8x32);
        for (uint8_t i_8row = 0; i_8row < 4; i_8row++)
            _mm256_storeu_si256(((__m256i *)out)+i_8row, out8x32[i_8row]);
        in += 32;
        in2 += 32;
        out += 32;
    }
}
#pragma GCC pop_options //-----------------------------------------------------

//______________________________________________________________________________
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

__m256i inline tbm_mult_t16x16_m256i_gfni(__m256i a, __m256i b)
{
    __m128i reverse8_2col = _mm_set_epi8(
        8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7);
    __m256i reverse8_col = _mm256_set_m128i(reverse8_2col, reverse8_2col);

    // We want to convert the 16x16 matrix to four 8x8 matrix
    // following the order: [[sub1, sub0], [sub3, sub2]]
    // first 16 bits are in least signicant bir order: b15 down to b0
    // odd and even octets represent submatrices sub1 and sub0
    // but as 16x16 matrix encodes a row as [b0, b1, ... b15]
    // matrix sub0 is actually located in the odd octets
    __m128i split8x8_2 = _mm_set_epi8(
        14, 12, 10, 8, 6, 4, 2, 0, 15, 13, 11, 9, 7, 5, 3, 1);
    __m256i split8x8_4 = _mm256_set_m128i(split8x8_2, split8x8_2);
    __m256i split8x8_4r = _mm256_shuffle_epi8(split8x8_4, reverse8_col);
    __m256i a3210 = _mm256_shuffle_epi8(a, split8x8_4);
    __m256i b3210r = _mm256_shuffle_epi8(b, split8x8_4r);

    __m256i a3311 = _mm256_permute4x64_epi64(a3210, _MM_SHUFFLE(3, 3, 1, 1));
    __m256i a2200 = _mm256_permute4x64_epi64(a3210, _MM_SHUFFLE(2, 2, 0, 0));
    __m256i b1313r = _mm256_permute4x64_epi64(b3210r, _MM_SHUFFLE(1, 3, 1, 3));
    __m256i b0202r = _mm256_permute4x64_epi64(b3210r, _MM_SHUFFLE(0, 2, 0, 2));

    __m256i out = _mm256_xor_si256(
        _mm256_gf2p8affine_epi64_epi8(a3311, b1313r, 0),
        _mm256_gf2p8affine_epi64_epi8(a2200, b0202r, 0));

    __m128i unsplit8x8_2 = _mm_set_epi8(
        7, 15, 6, 14, 5, 13, 4, 12, 3, 11, 2, 10, 1, 9, 0, 8);
    __m256i unsplit8x8_4 = _mm256_set_m128i(unsplit8x8_2, unsplit8x8_2);
    return _mm256_shuffle_epi8(out, unsplit8x8_4);
}

void inline tbm_mult_t32x32_m256i_gfni(__m256i a[4], __m256i b[4])
{
    // We want to convert the 8x32 matrix to four 8x8 matrix
    // following the order: [[sub1, sub0], [sub3, sub2]]
    // first 32 bits are in least signicant bir order: b31 down to b0
    // modulo-4 octets represent submatrices sub3 to sub0
    // but as 32x32 matrix encodes a row as [b0, b1, ... b31]
    // matrix sub0 is actually located in 3 modulo 4 octets
    __m128i split8x8_2 = _mm_set_epi8(
        12, 8, 4, 0, 13, 9, 5, 1, 14, 10, 6, 2, 15, 11, 7, 3);
    __m256i split8x8_4 = _mm256_set_m128i(split8x8_2, split8x8_2);
    __m256i split8x8_4_128 = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);

    // convert format for matrix a
    for (uint8_t i_row = 0; i_row < 4; i_row++)
    {
        __m256i a3210 = _mm256_shuffle_epi8(a[i_row], split8x8_4);
        a[i_row] = _mm256_permutevar8x32_epi32(a3210, split8x8_4_128);
    }
    
    __m128i split8x8_2r = _mm_set_epi8(
        0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    __m256i split8x8_4r = _mm256_set_m128i(split8x8_2r, split8x8_2r);
    __m256i split8x8_4r_128 = _mm256_set_epi32(3, 7, 2, 6, 1, 5, 0, 4);

    __m128i unsplit8x8_2 = _mm_set_epi8(
        3, 7, 11, 15, 2, 6, 10, 14, 1, 5, 9, 13, 0, 4, 8, 12);
    __m256i unsplit8x8_4 = _mm256_set_m128i(unsplit8x8_2, unsplit8x8_2);
    __m256i unsplit8x8_4_128 = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);

    __m256i b3210r = _mm256_shuffle_epi8(b[0], split8x8_4r);
    __m256i b7654r = _mm256_shuffle_epi8(b[1], split8x8_4r);
    __m256i bba98r = _mm256_shuffle_epi8(b[2], split8x8_4r);
    __m256i bfedcr = _mm256_shuffle_epi8(b[3], split8x8_4r);
    b3210r = _mm256_permutevar8x32_epi32(b3210r, split8x8_4r_128);
    b7654r = _mm256_permutevar8x32_epi32(b7654r, split8x8_4r_128);
    bba98r = _mm256_permutevar8x32_epi32(bba98r, split8x8_4r_128);
    bfedcr = _mm256_permutevar8x32_epi32(bfedcr, split8x8_4r_128);

    __m256i b32bar = _mm256_permute2x128_si256(b3210r, bba98r, 0x13);
    __m256i b76fer = _mm256_permute2x128_si256(b7654r, bfedcr, 0x13);
    __m256i b1098r = _mm256_permute2x128_si256(b3210r, bba98r, 0x02);
    __m256i b54dcr = _mm256_permute2x128_si256(b7654r, bfedcr, 0x02);

    __m256i b37bfr = _mm256_unpackhi_epi64 (b76fer, b32bar);
    __m256i b26aer = _mm256_unpacklo_epi64 (b76fer, b32bar);
    __m256i b159dr = _mm256_unpackhi_epi64 (b54dcr, b1098r);
    __m256i b048cr = _mm256_unpacklo_epi64 (b54dcr, b1098r);

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
        prod = _mm256_gf2p8affine_epi64_epi8(repeat, b048cr, 0);
        out = _mm256_xor_si256(out, prod);

        // repeat = _mm256_permute4x64_epi64(a3210, _MM_SHUFFLE(1, 1, 1, 1));
        repeat = _mm256_set1_epi64x(*a64++);
        prod = _mm256_gf2p8affine_epi64_epi8(repeat, b159dr, 0);
        out = _mm256_xor_si256(out, prod);

        // repeat = _mm256_permute4x64_epi64(a3210, _MM_SHUFFLE(2, 2, 2, 2));
        repeat = _mm256_set1_epi64x(*a64++);
        prod = _mm256_gf2p8affine_epi64_epi8(repeat, b26aer, 0);
        out = _mm256_xor_si256(out, prod);

        // repeat = _mm256_permute4x64_epi64(a3210, _MM_SHUFFLE(3, 3, 3, 3));
        repeat = _mm256_set1_epi64x(*a64++);
        prod = _mm256_gf2p8affine_epi64_epi8(repeat, b37bfr, 0);
        out = _mm256_xor_si256(out, prod);

        out = _mm256_permutevar8x32_epi32(out, unsplit8x8_4_128);
        out = _mm256_shuffle_epi8(out, unsplit8x8_4);
        a[i_row] = out;
    }
}

#pragma GCC push_options //----------------------------------------------------
#pragma GCC optimize("no-tree-vectorize")
void tbm_mult_t8x8_gfni(
    uint8_t *in, uint8_t *in2t, uint64_t n_mat, uint8_t *out)
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
#endif
    for (uint64_t i_mat = i_avx2; i_mat < n_mat; i_mat++)
    {
        *((uint64_t *)out) = tbm_mult_t8x8_uint64(*((uint64_t *)in), in2t);
        in += 8;
        in2t += 8;
        out += 8;
    }
}

void tbm_mult_t16x16_gfni(
    uint16_t *in, uint16_t *in2t, uint64_t n_mat, uint16_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        __m256i in16x16 = _mm256_loadu_si256((__m256i *)in);
        __m256i in2_16x16 = _mm256_loadu_si256((__m256i *)in2t);
        __m256i out16x16 = tbm_mult_t16x16_m256i_gfni(in16x16, in2_16x16);
        _mm256_storeu_si256((__m256i *)out, out16x16);
        in += 16;
        in2t += 16;
        out += 16;
    }
}

void tbm_mult_t32x32_gfni(
    uint32_t *in, uint32_t *in2t, uint64_t n_mat, uint32_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        __m256i in8x32[4];
        __m256i in2t_8x32[4];
        for (uint8_t i_8row = 0; i_8row < 4; i_8row++)
        {
            in8x32[i_8row] = _mm256_loadu_si256(((__m256i *)in)+i_8row);
            in2t_8x32[i_8row] = _mm256_loadu_si256(((__m256i *)in2t)+i_8row);
        }
        tbm_mult_t32x32_m256i_gfni(in8x32, in2t_8x32);
        for (uint8_t i_8row = 0; i_8row < 4; i_8row++)
            _mm256_storeu_si256(((__m256i *)out)+i_8row, in8x32[i_8row]);
        in += 32;
        in2t += 32;
        out += 32;
}
}
#pragma GCC pop_options //-----------------------------------------------------
