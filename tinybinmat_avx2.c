#include "immintrin.h"
#include "tinybinmat.h"
#include "tinybinmat_template.c"

#if defined(USE_GFNI)
#define __SUFFIX(fun) fun##_gfni
#else
#define __SUFFIX(fun) fun##_avx2

void print_avx2_uint64(__m256i reg)
{
    uint64_t *ptr = (uint64_t *)&reg;
    for (uint8_t k = 3; k != 0; k--)
        printf("%016lx ", ptr[k]);
    printf("%016lx\n", ptr[0]);
}

__m256i _mm256_movm_epi8_avx2(const uint32_t mask)
{
    __m256i vmask = _mm256_set1_epi32(mask);
    const __m256i shuffle = _mm256_set_epi64x(
        0x0303030303030303, 0x0202020202020202,
        0x0101010101010101, 0x0000000000000000);
    vmask = _mm256_shuffle_epi8(vmask, shuffle);
    // "%016x" % (0x7fbfdfeff7fbfdfe ^ ((2 << 64)-1)) -> '18040201008040201'
    const __m256i bit_mask = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe);
    vmask = _mm256_or_si256(vmask, bit_mask);
    return _mm256_cmpeq_epi8(vmask, _mm256_set1_epi64x(-1));
}
#endif

//______________________________________________________________________________
#if defined(USE_GFNI)
static __m256i inline tbm_transpose8x8_m256i(__m256i in8x8_4)
{
    // _mm256_gf2p8affine_epi64_epi8(I, A, 0) is (A*I.T).T = A.T
    __m256i eye_8x8_4 = _mm256_set1_epi64x(0x0102040810204080);
    return _mm256_gf2p8affine_epi64_epi8(eye_8x8_4, in8x8_4, 0);
}
#else
static __m256i inline tbm_transpose8x8_m256i(__m256i in8x8_4)
{
    __m256i ur_mask4x4 = _mm256_set1_epi64x(0xf0f0f0f000000000);
    __m256i xor = _mm256_xor_si256(in8x8_4, _mm256_slli_epi64(in8x8_4, 36));
    xor = _mm256_and_si256(xor, ur_mask4x4);
    in8x8_4 = _mm256_xor_si256(in8x8_4, xor);
    xor = _mm256_srli_epi64(xor, 36);
    in8x8_4 = _mm256_xor_si256(in8x8_4, xor);
    __m256i ur_mask2x2 = _mm256_set1_epi64x(0xcccc0000cccc0000);
    xor = _mm256_xor_si256(in8x8_4, _mm256_slli_epi64(in8x8_4, 18));
    xor = _mm256_and_si256(xor, ur_mask2x2);
    in8x8_4 = _mm256_xor_si256(in8x8_4, xor);
    xor = _mm256_srli_epi64(xor, 18);
    in8x8_4 = _mm256_xor_si256(in8x8_4, xor);
    __m256i ur_mask1x1 = _mm256_set1_epi64x(0xaa00aa00aa00aa00);
    xor = _mm256_xor_si256(in8x8_4, _mm256_slli_epi64(in8x8_4, 9));
    xor = _mm256_and_si256(xor, ur_mask1x1);
    in8x8_4 = _mm256_xor_si256(in8x8_4, xor);
    xor = _mm256_srli_epi64(xor, 9);
    in8x8_4 = _mm256_xor_si256(in8x8_4, xor);
    return in8x8_4;
}
#endif

static void inline tbm_transpose8x8_x4(uint64_t in[4])
{
    __m256i in8x8_4 = _mm256_loadu_si256((__m256i *)in);
    _mm256_storeu_si256((__m256i *)in, tbm_transpose8x8_m256i(in8x8_4));
}

static __attribute__ ((noinline)) void tbm_transpose_1d(
    uint64_t *in, uint64_t n8x8, uint64_t *out)
{
    uint64_t i8x8; //!< index for 4 8x8 blocks
    for (i8x8 = 0; i8x8 < n8x8/4*4; i8x8 += 4)
    {
        // load 4x8x8 blocks
        __m256i in8x8_4 = _mm256_loadu_si256((__m256i *)(in+i8x8));
        // transpose 4x8x8 blocks
        __m256i out8x8_4 = tbm_transpose8x8_m256i(in8x8_4);
        // store transposed 4x8x8 blocks
        _mm256_storeu_si256((__m256i *)(out+i8x8), out8x8_4);
    }
    if (i8x8 == n8x8)
        return; // all blocks are processed
    __m256i in8x8_4 = _mm256_loadu_si256((__m256i *)(in+i8x8));
    __m256i out8x8_4 = tbm_transpose8x8_m256i(in8x8_4);
    __m256i mask = _mm256_set_epi64x(3, 2, 1, 0); //!< mask for the last block
    mask = _mm256_cmpgt_epi64(_mm256_set1_epi64x(n8x8-i8x8), mask);
    _mm256_maskstore_epi64((long long int *)(out+i8x8), mask, out8x8_4);
}

static __attribute__ ((noinline)) void tbm_transpose_2x2(
    __m256i *in, uint64_t n_mat, __m256i *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        // load 4x8x8 blocks
        __m256i in8x8_4 = _mm256_loadu_si256(in+i_mat);
        // transpose 4x8x8 blocks
        __m256i a3210r = tbm_transpose8x8_m256i(in8x8_4);
        __m256i a3120r = _mm256_permute4x64_epi64(
            a3210r, _MM_SHUFFLE(3, 1, 2, 0));

        // store transposed 4x8x8 blocks
        _mm256_storeu_si256(out+i_mat, a3120r);
    }
}

static __attribute__ ((noinline)) void tbm_transpose_4x4(
    __m256i *in, uint64_t n_mat, __m256i *out)
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

        _mm256_storeu_si256(out+i_mat*4+0, tbm_transpose8x8_m256i(inc840));
        _mm256_storeu_si256(out+i_mat*4+1, tbm_transpose8x8_m256i(ind951));
        _mm256_storeu_si256(out+i_mat*4+2, tbm_transpose8x8_m256i(inea62));
        _mm256_storeu_si256(out+i_mat*4+3, tbm_transpose8x8_m256i(infb73));
    }
}

#pragma GCC push_options //----------------------------------------------------
#pragma GCC optimize("no-tree-vectorize")
static void __attribute__ ((noinline)) tbm_transpose_256(
    uint64_t *in, uint64_t n_mat, uint32_t n_row8, uint32_t n_col8,
    uint64_t *out)
{
    tbm_transpose_256_template(
        in, n_mat, n_row8, n_col8, out, tbm_transpose8x8_x4);
}

#pragma GCC pop_options //-----------------------------------------------------

void __SUFFIX(tbm_transpose) (
    uint64_t *in, uint64_t n_mat, uint32_t n_row8, uint32_t n_col8,
    uint64_t *out)
{
    if (n_row8 == 1 || n_col8 == 1)
    {
        // no block transpose needed
        uint64_t n8x8 = n_mat*n_row8*n_col8; //!< number of 8x8 blocks
        return tbm_transpose_1d(in, n8x8, out);
    }
    else if (n_row8 == 2 && n_col8 == 2)
    {
        return tbm_transpose_2x2((__m256i *)in, n_mat, (__m256i *)out);
    }
    else if (n_row8 == 4 && n_col8 == 4)
    {
        return tbm_transpose_4x4((__m256i *)in, n_mat, (__m256i *)out);
    }
    tbm_transpose_256(in, n_mat, n_row8, n_col8, out);
}

//______________________________________________________________________________
#if defined(USE_GFNI)
static __m256i inline tbm_mult8x8_m256i(__m256i a, __m256i b)
{
    // _mm256_gf2p8affine_epi64_epi8(B, A, 0) is (A*B.T).T
    // _mm256_gf2p8affine_epi64_epi8(A, B.T, 0) is (B.T*A.T).T = A*B
    return _mm256_gf2p8affine_epi64_epi8(a, tbm_transpose8x8_m256i(b), 0);
}
#else
static __m256i inline tbm_mult8x8_m256i(__m256i a, __m256i b)
{
    __m128i repeat8x2 = _mm_set_epi8(
        8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0);
    __m256i repeat8x4 = _mm256_set_m128i(repeat8x2, repeat8x2);

    __m256i out = _mm256_setzero_si256();
    __m256i test_bit = _mm256_set1_epi8(128);
    for (uint8_t i_bit = 0; i_bit < 8; i_bit++)
    {
        // create bit mask from the most significant bits in a octets
        __m256i bit_a = _mm256_and_si256(a, test_bit);
        bit_a = _mm256_cmpeq_epi8(bit_a, test_bit);
        a = _mm256_slli_epi64(a, 1);
        // repeat 8 times least significant octet of b 8x8 matrices
        __m256i b_repeat = _mm256_shuffle_epi8(b, repeat8x4);
        b = _mm256_srli_epi64(b, 8);
        __m256i prod = _mm256_and_si256(bit_a, b_repeat);
        out = _mm256_xor_si256(out, prod);
    }
    return out;
}
#endif

static void inline tbm_mult8x8_1x4(
    uint64_t a, uint64_t b[4], uint64_t out[4])
{
    __m256i _a = _mm256_set1_epi64x(a);
    __m256i _b = _mm256_loadu_si256((__m256i *)b);
    _mm256_storeu_si256((__m256i *)out, tbm_mult8x8_m256i(_a, _b));
}

static void inline tbm_mult8x8_x4(
    uint64_t a[4], uint64_t b[4], uint64_t out[4])
{
    __m256i _a = _mm256_loadu_si256((__m256i *)a);
    __m256i _b = _mm256_loadu_si256((__m256i *)b);
    _mm256_storeu_si256((__m256i *)out, tbm_mult8x8_m256i(_a, _b));
}

static __m256i inline tbm_mult16x16_m256i(__m256i a3210, __m256i b3210)
{
    // a is:    b is:
    // [0, 1]   [0, 1]   [0, 0]   [0, 1]   [1, 1]   [2, 3]
    // [2, 3] @ [2, 3] = [2, 2] * [0, 1] + [3, 3] * [2, 3]
    __m256i a3311 = _mm256_permute4x64_epi64(a3210, _MM_SHUFFLE(3, 3, 1, 1));
    __m256i a2200 = _mm256_permute4x64_epi64(a3210, _MM_SHUFFLE(2, 2, 0, 0));
    __m256i b3232 = _mm256_permute4x64_epi64(b3210, _MM_SHUFFLE(3, 2, 3, 2));
    __m256i b1010 = _mm256_permute4x64_epi64(b3210, _MM_SHUFFLE(1, 0, 1, 0));

    return _mm256_xor_si256(
        tbm_mult8x8_m256i(a3311, b3232), tbm_mult8x8_m256i(a2200, b1010));
}

static void inline tbm_mult32x32_m256i(__m256i a[4], __m256i b[4])
{
    uint64_t *a64 = (uint64_t *)a;
    for (uint8_t i_row = 0; i_row < 4; i_row++)
    {
        __m256i repeat; //<! current value of a cell 4 times
        __m256i out; //<! accumulated sum of the products
        out = _mm256_setzero_si256();
        repeat = _mm256_set1_epi64x(*a64++);
        out = _mm256_xor_si256(out, tbm_mult8x8_m256i(repeat, b[0]));

        repeat = _mm256_set1_epi64x(*a64++);
        out = _mm256_xor_si256(out, tbm_mult8x8_m256i(repeat, b[1]));

        repeat = _mm256_set1_epi64x(*a64++);
        out = _mm256_xor_si256(out, tbm_mult8x8_m256i(repeat, b[2]));

        repeat = _mm256_set1_epi64x(*a64++);
        out = _mm256_xor_si256(out, tbm_mult8x8_m256i(repeat, b[3]));

        a[i_row] = out;
    }
}

#pragma GCC push_options //-----------------------------------------------------
#pragma GCC optimize("no-tree-vectorize")
static void __attribute__ ((noinline)) tbm_mult_256(
    uint64_t *in, uint64_t n_mat, uint32_t n_row8, uint32_t n_col8,
    uint64_t *in2, uint32_t n_col8_2, uint64_t *out)
{
    tbm_mult_256_template(
        in, n_mat, n_row8, n_col8, in2, n_col8_2, out, tbm_mult8x8_1x4);
}

static void __attribute__ ((noinline)) tbm_mult_ncol8_1(
    uint64_t *in, uint64_t n_mat, uint64_t *in2, uint64_t *out)
{
    uint64_t i8x8; //!< index for 4 8x8 blocks
    for (i8x8 = 0; i8x8 < n_mat/4*4; i8x8 += 4)
    {
        __m256i in8x8_4 = _mm256_loadu_si256((__m256i *)(in+i8x8));
        __m256i in2_8x8_4 = _mm256_loadu_si256((__m256i *)(in2+i8x8));
        _mm256_storeu_si256(
            (__m256i *)(out+i8x8), tbm_mult8x8_m256i(in8x8_4, in2_8x8_4));
    }

    if (i8x8 == n_mat)
        return; // all blocks are processed
    __m256i in8x8_4 = _mm256_loadu_si256((__m256i *)(in+i8x8));
    __m256i in2_8x8_4 = _mm256_loadu_si256((__m256i *)(in2+i8x8));
    __m256i mask = _mm256_set_epi64x(3, 2, 1, 0); //!< mask for the last block
    mask = _mm256_cmpgt_epi64(_mm256_set1_epi64x(n_mat-i8x8), mask);
    _mm256_maskstore_epi64(
        (long long int *)(out+i8x8), mask,
        tbm_mult8x8_m256i(in8x8_4, in2_8x8_4));
}

static void __attribute__ ((noinline)) tbm_mult_ncol8_2(
    uint64_t *in, uint64_t n_mat, uint64_t *in2, uint64_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        __m256i in16X16 = _mm256_loadu_si256((__m256i *)in);
        __m256i in2_16X16 = _mm256_loadu_si256((__m256i *)in2);
        _mm256_storeu_si256(
            (__m256i *)out, tbm_mult16x16_m256i(in16X16, in2_16X16));
        in += 4;
        in2 += 4;
        out += 4;
    }
}

#if 1 // this code is faster than the one below
static void __attribute__ ((noinline)) tbm_mult_ncol8_4(
    uint64_t *in, uint64_t n_mat, uint64_t *in2, uint64_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        __m256i in8x32[4];
        __m256i in2_8x32[4];
        for (uint8_t i_8row = 0; i_8row < 4; i_8row++)
        {
            in8x32[i_8row] = _mm256_loadu_si256(((__m256i *)in)+i_8row);
            in2_8x32[i_8row] = _mm256_loadu_si256(((__m256i *)in2)+i_8row);
        }
        tbm_mult32x32_m256i(in8x32, in2_8x32);
        for (uint8_t i_8row = 0; i_8row < 4; i_8row++)
            _mm256_storeu_si256(((__m256i *)out)+i_8row, in8x32[i_8row]);
        in += 16;
        in2 += 16;
        out += 16;
    }
}
#else
static void __attribute__ ((noinline)) tbm_mult_ncol8_4(
    uint64_t *in, uint64_t n_mat, uint64_t *in2, uint64_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        tbm_mult32x32_template(in, in2, out, tbm_mult8x8_1x4);
        in += 16;
        in2 += 16;
        out += 16;
    }
}
#endif

#pragma GCC pop_options //------------------------------------------------------

void __SUFFIX(tbm_mult) (
    uint64_t *in, uint64_t n_mat, uint32_t n_row8, uint32_t n_col8,
    uint64_t *in2, uint32_t n_col8_2, uint64_t *out)
{
    if ((n_col8 == 1) && (n_row8 == 1) && (n_col8_2 == 1))
        return tbm_mult_ncol8_1(in, n_mat, in2, out);
    if ((n_col8 == 2) && (n_row8 == 2) && (n_col8_2 == 2))
        return tbm_mult_ncol8_2(in, n_mat, in2, out);
    if ((n_col8 == 4) && (n_row8 == 4) && (n_col8_2 == 4))
        return tbm_mult_ncol8_4(in, n_mat, in2, out);

    tbm_mult_256(in, n_mat, n_row8, n_col8, in2, n_col8_2, out);
}

//______________________________________________________________________________
#if defined(USE_GFNI)
static __m256i inline tbm_mult_t8x8_m256i(__m256i a, __m256i b)
{
    // _mm256_gf2p8affine_epi64_epi8(B, A, 0) is (A*B.T).T
    // _mm256_gf2p8affine_epi64_epi8(A, B, 0) is (B*A.T).T = A*B.T
    return _mm256_gf2p8affine_epi64_epi8(a, b, 0);
}
#else
// non-functional!
uint64_t inline tbm_mult_t8x8_dot_avx2(uint64_t a8x8, uint64_t tb8x8)
{
    __m256i tb8x8_4 = _mm256_set1_epi64x(tb8x8);
    __m256i row_a_4 = _mm256_cvtepu8_epi64(_mm_set_epi64x(0, a8x8 >> 32));
    __m128i repeat8x2 = _mm_set_epi8(
        8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0);
    __m256i repeat8x4 = _mm256_set_m128i(repeat8x2, repeat8x2);

    __m256i repeat_4 = _mm256_shuffle_epi8(row_a_4, repeat8x4);
    __m256i prod_4 = _mm256_and_si256(tb8x8_4, repeat_4);
    prod_4 = _mm256_xor_si256(prod_4, _mm256_slli_epi16(prod_4, 4));
    prod_4 = _mm256_xor_si256(prod_4, _mm256_slli_epi16(prod_4, 2));
    prod_4 = _mm256_xor_si256(prod_4, _mm256_slli_epi16(prod_4, 1));
    uint64_t out = (uint32_t)_mm256_movemask_epi8(prod_4);

    row_a_4 = _mm256_cvtepu8_epi64(_mm_set_epi64x(0, a8x8));
    repeat_4 = _mm256_shuffle_epi8(row_a_4, repeat8x4);
    prod_4 = _mm256_and_si256(tb8x8_4, repeat_4);
    prod_4 = _mm256_xor_si256(prod_4, _mm256_slli_epi16(prod_4, 4));
    prod_4 = _mm256_xor_si256(prod_4, _mm256_slli_epi16(prod_4, 2));
    prod_4 = _mm256_xor_si256(prod_4, _mm256_slli_epi16(prod_4, 1));
    out <<= 32;
    out |= (uint32_t)_mm256_movemask_epi8(prod_4);
    return out;
}

static __m256i inline tbm_mult_t8x8_m256i(__m256i a, __m256i b)
{
    return tbm_mult8x8_m256i(a, tbm_transpose8x8_m256i(b));
}
#endif

static uint64_t inline tbm_dot_t(uint64_t *in, uint64_t *in2, uint32_t n_dot)
{
    uint64_t i8x8; //!< index for 4 8x8 blocks
    uint64_t n8x8 = n_dot; //!< number of 8x8 blocks
    __m256i in8x8_4; //!< 4 8x8 blocks from the first matrix
    __m256i in2_8x8_4; //!< 4 8x8 blocks from the second transposed matrix
__m256i acc = _mm256_setzero_si256(); //!< accumulator for 4 8x8 blocks
    for (i8x8 = 0; i8x8 < n8x8/4*4; i8x8 += 4)
    {
        // load 4x8x8 blocks
        in8x8_4 = _mm256_loadu_si256((__m256i *)(in+i8x8));
        in2_8x8_4 = _mm256_loadu_si256((__m256i *)(in2+i8x8));
        // transpose 4x8x8 blocks
        __m256i out8x8_4 = tbm_mult_t8x8_m256i(in8x8_4, in2_8x8_4);
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
        __m256i out8x8_4 = tbm_mult_t8x8_m256i(in8x8_4, in2_8x8_4);
        acc = _mm256_xor_si256(acc, out8x8_4);
    }
    __m128i acc128 = _mm256_extracti128_si256(acc, 0);
    acc128 = _mm_xor_si128(acc128, _mm256_extracti128_si256(acc, 1));
    return _mm_extract_epi64(acc128, 0) ^ _mm_extract_epi64(acc128, 1);
}

static __m256i inline tbm_mult_t16x16_m256i(__m256i a3210, __m256i b3210)
{
    // a is:    b.T is:
    // [0, 1]   [0, 2]   [0, 0]   [0, 2]   [1, 1]   [1, 3]
    // [2, 3] @ [1, 3] = [2, 2] * [0, 2] + [3, 3] * [1, 3]
    __m256i a3311 = _mm256_permute4x64_epi64(a3210, _MM_SHUFFLE(3, 3, 1, 1));
    __m256i a2200 = _mm256_permute4x64_epi64(a3210, _MM_SHUFFLE(2, 2, 0, 0));
    __m256i b3131 = _mm256_permute4x64_epi64(b3210, _MM_SHUFFLE(3, 1, 3, 1));
    __m256i b2020 = _mm256_permute4x64_epi64(b3210, _MM_SHUFFLE(2, 0, 2, 0));

    return _mm256_xor_si256(
        tbm_mult_t8x8_m256i(a3311, b3131), tbm_mult_t8x8_m256i(a2200, b2020));
}

static void inline tbm_mult_t32x32_m256i(__m256i a[4], __m256i b[4])
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
        prod = tbm_mult_t8x8_m256i(repeat, b_c840);
        out = _mm256_xor_si256(out, prod);

        // repeat = _mm256_permute4x64_epi64(a3210, _MM_SHUFFLE(1, 1, 1, 1));
        repeat = _mm256_set1_epi64x(*a64++);
        prod = tbm_mult_t8x8_m256i(repeat, b_d951);
        out = _mm256_xor_si256(out, prod);

        // repeat = _mm256_permute4x64_epi64(a3210, _MM_SHUFFLE(2, 2, 2, 2));
        repeat = _mm256_set1_epi64x(*a64++);
        prod = tbm_mult_t8x8_m256i(repeat, b_ea62);
        out = _mm256_xor_si256(out, prod);

        // repeat = _mm256_permute4x64_epi64(a3210, _MM_SHUFFLE(3, 3, 3, 3));
        repeat = _mm256_set1_epi64x(*a64++);
        prod = tbm_mult_t8x8_m256i(repeat, b_fb73);
        out = _mm256_xor_si256(out, prod);

        a[i_row] = out;
    }
}

#pragma GCC push_options //-----------------------------------------------------
#pragma GCC optimize("no-tree-vectorize")
static void __attribute__ ((noinline)) tbm_mult_t_dot(
    uint64_t *in, uint64_t n_mat, uint32_t n_row8, uint32_t n_col8,
    uint64_t *in2, uint32_t n_col8_2, uint64_t *out)
{
    tbm_mult_t_dot_template(
        in, n_mat, n_row8, n_col8, in2, n_col8_2, out, tbm_dot_t);
}

static void __attribute__ ((noinline)) tbm_mult_t_ncol8_1(
    uint64_t *in, uint64_t n_mat, uint64_t *in2, uint64_t *out)
{
#if defined(USE_DOT)
    tbm_mult_t_dot_(in, n_mat, 1, 1, in2, 1, out);
#else
    uint64_t i8x8; //!< index for 4 8x8 blocks
    for (i8x8 = 0; i8x8 < n_mat/4*4; i8x8 += 4)
    {
        __m256i in8x8_4 = _mm256_loadu_si256((__m256i *)(in+i8x8));
        __m256i in2_8x8_4 = _mm256_loadu_si256((__m256i *)(in2+i8x8));
        _mm256_storeu_si256(
            (__m256i *)(out+i8x8),
            tbm_mult_t8x8_m256i(in8x8_4, in2_8x8_4));
    }

    if (i8x8 == n_mat)
        return; // all blocks are processed
    __m256i in8x8_4 = _mm256_loadu_si256((__m256i *)(in+i8x8));
    __m256i in2_8x8_4 = _mm256_loadu_si256((__m256i *)(in2+i8x8));
    __m256i mask = _mm256_set_epi64x(3, 2, 1, 0); //!< mask for the last block
    mask = _mm256_cmpgt_epi64(_mm256_set1_epi64x(n_mat-i8x8), mask);
    _mm256_maskstore_epi64(
        (long long int *)(out+i8x8), mask,
        tbm_mult_t8x8_m256i(in8x8_4, in2_8x8_4));
#endif
}

static void __attribute__ ((noinline)) tbm_mult_t_ncol8_2(
    uint64_t *in, uint64_t n_mat, uint64_t *in2, uint64_t *out)
{
#if defined(USE_DOT)
    tbm_mult_t_dot(in, n_mat, 2, 2, in2, 2, out);
#else
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        __m256i in16X16 = _mm256_loadu_si256((__m256i *)in);
        __m256i in2_16X16 = _mm256_loadu_si256((__m256i *)in2);
        _mm256_storeu_si256(
            (__m256i *)out, tbm_mult_t16x16_m256i(in16X16, in2_16X16));
        in += 4;
        in2 += 4;
        out += 4;
    }
#endif
}

static void __attribute__ ((noinline)) tbm_mult_t_ncol8_4(
    uint64_t *in, uint64_t n_mat, uint64_t *in2, uint64_t *out)
{
#if defined(USE_DOT)
    tbm_mult_t_dot(in, n_mat, 4, 4, in2, 4, out);
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
        tbm_mult_t32x32_m256i(in8x32, in2_8x32);
        for (uint8_t i_8row = 0; i_8row < 4; i_8row++)
            _mm256_storeu_si256(((__m256i *)out)+i_8row, in8x32[i_8row]);
        in += 16;
        in2 += 16;
        out += 16;
    }
#endif
}

#pragma GCC pop_options //------------------------------------------------------

void __SUFFIX(tbm_mult_t) (
    uint64_t *in, uint64_t n_mat, uint32_t n_row8, uint32_t n_col8,
    uint64_t *in2, uint32_t n_row8_2, uint64_t *out)
{
    if ((n_col8 == 1) && (n_row8 == 1) && (n_row8_2 == 1))
        return tbm_mult_t_ncol8_1(in, n_mat, in2, out);
    if ((n_col8 == 2) && (n_row8 == 2) && (n_row8_2 == 2))
        return tbm_mult_t_ncol8_2(in, n_mat, in2, out);
    if ((n_col8 == 4) && (n_row8 == 4) && (n_row8_2 == 4))
        return tbm_mult_t_ncol8_4(in, n_mat, in2, out);

    tbm_mult_t_dot(in, n_mat, n_row8, n_col8, in2, n_row8_2, out);
}
