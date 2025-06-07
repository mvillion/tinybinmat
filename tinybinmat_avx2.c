#include "tinybinmat.h"
#include "tinybinmat_template.c"

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

//______________________________________________________________________________
__m256i inline inline tbm_transpose8x8_m256i_avx2(__m256i in8x8_4)
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

void inline inline tbm_transpose8x8_x4_avx2(uint64_t in[4])
{
    __m256i in8x8_4 = _mm256_loadu_si256((__m256i *)in);
    _mm256_storeu_si256((__m256i *)in, tbm_transpose8x8_m256i_avx2(in8x8_4));
}

void __attribute__ ((noinline)) tbm_transpose_avx2_1d(
    uint64_t *in, uint64_t n_mat, uint64_t *out)
{
    uint64_t i_mat;
    uint64_t tmp[4]; //!< temporary storage for 4 8x8 blocks
    for (i_mat = 0; i_mat < n_mat/4*4; i_mat += 4)
    {
        for (uint8_t i_4 = 0; i_4 < 4; i_4++)
            out[i_mat+i_4] = in[i_mat+i_4];
        tbm_transpose8x8_x4_avx2(out+i_mat);
    }
    if (i_mat == n_mat)
        return; // all blocks are processed
    for (uint8_t i_4 = 0; i_4 < (n_mat & 3); i_4++)
        tmp[i_4] = in[i_mat+i_4];
    tbm_transpose8x8_x4_avx2(tmp);
    for (uint8_t i_4 = 0; i_4 < (n_mat & 3); i_4++)
        out[i_mat+i_4] = tmp[i_4];
}

static __attribute__ ((noinline)) void tbm_transpose_avx2_2x2(
    __m256i *in, uint64_t n_mat, __m256i *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        // load 4x8x8 blocks
        __m256i in8x8_4 = _mm256_loadu_si256(in+i_mat);
        // transpose 4x8x8 blocks
        __m256i a3210r = tbm_transpose8x8_m256i_avx2(in8x8_4);
        __m256i a3120r = _mm256_permute4x64_epi64(
            a3210r, _MM_SHUFFLE(3, 1, 2, 0));
    
        // store transposed 4x8x8 blocks
        _mm256_storeu_si256(out+i_mat, a3120r);
    }
}

static __attribute__ ((noinline)) void tbm_transpose_avx2_4x4(
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

        _mm256_storeu_si256(out+i_mat*4+0, tbm_transpose8x8_m256i_avx2(inc840));
        _mm256_storeu_si256(out+i_mat*4+1, tbm_transpose8x8_m256i_avx2(ind951));
        _mm256_storeu_si256(out+i_mat*4+2, tbm_transpose8x8_m256i_avx2(inea62));
        _mm256_storeu_si256(out+i_mat*4+3, tbm_transpose8x8_m256i_avx2(infb73));
    }
}

#pragma GCC push_options //----------------------------------------------------
#pragma GCC optimize("no-tree-vectorize")
void tbm_transpose_avx2_256(
    uint64_t *in, uint64_t n_mat, uint32_t n_row8, uint32_t n_col8, 
    uint64_t *out)
{
    tbm_transpose_256_template(
        in, n_mat, n_row8, n_col8, out, tbm_transpose8x8_x4_avx2);
}

#pragma GCC pop_options //-----------------------------------------------------

void tbm_transpose_avx2(
    uint64_t *in, uint64_t n_mat, uint32_t n_row8, uint32_t n_col8, 
    uint64_t *out)
{
    if (n_row8 == 1 || n_col8 == 1)
    {
        // no block transpose needed
        uint64_t n8x8 = n_mat*n_row8*n_col8; //!< number of 8x8 blocks
        return tbm_transpose_avx2_1d(in, n8x8, out);
    }
    else if (n_row8 == 2 && n_col8 == 2)
    {
        return tbm_transpose_avx2_2x2((__m256i *)in, n_mat, (__m256i *)out);
    }
    else if (n_row8 == 4 && n_col8 == 4)
    {
        return tbm_transpose_avx2_4x4((__m256i *)in, n_mat, (__m256i *)out);
    }
    tbm_transpose_avx2_256(in, n_mat, n_row8, n_col8, out);
}

//______________________________________________________________________________
__m256i inline tbm_mult8x8_m256i_avx2(__m256i a, uint8_t *b8)
{
    __m128i repeat8x2 = _mm_set_epi8(
        8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0);
    __m256i repeat8x4 = _mm256_set_m128i(repeat8x2, repeat8x2);
    
    __m256i out = _mm256_setzero_si256();
    __m256i test_bit = _mm256_set1_epi8(1);
    for (uint8_t i_bit = 0; i_bit < 8; i_bit++)
    {
        // create bit mask from the least significant bits in a
        __m256i bit_a = _mm256_and_si256(a, test_bit);
        bit_a = _mm256_cmpeq_epi8(bit_a, test_bit);
        test_bit = _mm256_add_epi8(test_bit, test_bit);
        // load 32 octets from b starting from i_bit
        __m256i b_i_bit32 = _mm256_loadu_si256((__m256i *)(b8+7-i_bit));
        b_i_bit32 = _mm256_shuffle_epi8(b_i_bit32, repeat8x4);
        __m256i prod = _mm256_and_si256(bit_a, b_i_bit32);
        out = _mm256_xor_si256(out, prod);
    }
    return out;
}

void inline tbm_mult8x8_1x4_avx2(uint64_t a, uint64_t b[4], uint64_t out[4])
{
    __m256i _a = _mm256_set1_epi64x(a);
    uint8_t *_b = (uint8_t *)b;
    _mm256_storeu_si256((__m256i *)out, tbm_mult8x8_m256i_avx2(_a, _b)); 
}

void inline tbm_mult8x8_x4_avx2(uint64_t a[4], uint64_t b[4], uint64_t out[4])
{
    __m256i _a = _mm256_loadu_si256((__m256i *)a);
    uint8_t *_b = (uint8_t *)b;
    _mm256_storeu_si256((__m256i *)out, tbm_mult8x8_m256i_avx2(_a, _b)); 
}

__m256i inline tbm_mult16x16_m256i_avx2(__m256i a3210, __m256i b3210)
{
    // a is:    b is:
    // [0, 1]   [0, 1]   [0, 0]   [0, 1]   [1, 1]   [2, 3]
    // [2, 3] @ [2, 3] = [2, 2] * [0, 1] + [3, 3] * [2, 3]
    __m256i a3311 = _mm256_permute4x64_epi64(a3210, _MM_SHUFFLE(3, 3, 1, 1));
    __m256i a2200 = _mm256_permute4x64_epi64(a3210, _MM_SHUFFLE(2, 2, 0, 0));
    __m256i b3232 = _mm256_permute4x64_epi64(b3210, _MM_SHUFFLE(3, 2, 3, 2));
    __m256i b1010 = _mm256_permute4x64_epi64(b3210, _MM_SHUFFLE(1, 0, 1, 0));
    uint8_t b1010_8[32+7]; // 7 octets for loadu access
    _mm256_storeu_si256((__m256i *)b1010_8, b1010);
    uint8_t b3232_8[32+7];
    _mm256_storeu_si256((__m256i *)b3232_8, b3232);


    __m256i out = _mm256_xor_si256(
        tbm_mult8x8_m256i_avx2(a3311, b3232_8),
        tbm_mult8x8_m256i_avx2(a2200, b1010_8));

    return out;
}

void tbm_mult_avx2_256(
    uint64_t *in, uint64_t n_mat, uint32_t n_row8, uint32_t n_col8,
    uint64_t *in2, uint32_t n_col8_2, uint64_t *out)
{
    tbm_mult_256_template(
        in, n_mat, n_row8, n_col8, in2, n_col8_2, out, tbm_mult8x8_1x4_avx2);
}

static void __attribute__ ((noinline)) tbm_mult_avx2_ncol8_1(
    uint64_t *in, uint64_t n_mat, uint64_t *in2, uint64_t *out)
{
    uint64_t i8x8; //!< index for 4 8x8 blocks
    for (i8x8 = 0; i8x8 < n_mat/4*4; i8x8 += 4)
    {
        __m256i in8x8_4 = _mm256_loadu_si256((__m256i *)(in+i8x8));
        uint8_t *in2_8x8_4 = (uint8_t *)(in2+i8x8);
        _mm256_storeu_si256(
            (__m256i *)(out+i8x8),
            tbm_mult8x8_m256i_avx2(in8x8_4, in2_8x8_4));
    }

    if (i8x8 == n_mat)
        return; // all blocks are processed
    __m256i in8x8_4 = _mm256_loadu_si256((__m256i *)(in+i8x8));
    uint8_t *in2_8x8_4 = (uint8_t *)(in2+i8x8);
    __m256i mask = _mm256_set_epi64x(3, 2, 1, 0); //!< mask for the last block
    mask = _mm256_cmpgt_epi64(_mm256_set1_epi64x(n_mat-i8x8), mask);
    _mm256_maskstore_epi64(
        (long long int *)(out+i8x8), mask, 
        tbm_mult8x8_m256i_avx2(in8x8_4, in2_8x8_4));
}

static void __attribute__ ((noinline)) tbm_mult_avx2_ncol8_2(
    uint64_t *in, uint64_t n_mat, uint64_t *in2, uint64_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        __m256i in16X16 = _mm256_loadu_si256((__m256i *)in);
        __m256i in2_16X16 = _mm256_loadu_si256((__m256i *)in2);
        _mm256_storeu_si256(
            (__m256i *)out, tbm_mult16x16_m256i_avx2(in16X16, in2_16X16));
        in += 4;
        in2 += 4;
        out += 4;
    }
}

static void __attribute__ ((noinline)) tbm_mult_avx2_ncol8_4(
    uint64_t *in, uint64_t n_mat, uint64_t *in2, uint64_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        tbm_mult32x32_template(in, in2, out, tbm_mult8x8_1x4_avx2);
        in += 16;
        in2 += 16;
        out += 16;
    }
}

void tbm_mult_avx2(
    uint64_t *in, uint64_t n_mat, uint32_t n_row8, uint32_t n_col8,
    uint64_t *in2, uint32_t n_col8_2, uint64_t *out)
{
    if ((n_col8 == 1) && (n_row8 == 1) && (n_col8_2 == 1))
        return tbm_mult_avx2_ncol8_1(in, n_mat, in2, out);
    if ((n_col8 == 2) && (n_row8 == 2) && (n_col8_2 == 2))
        return tbm_mult_avx2_ncol8_2(in, n_mat, in2, out);
    if ((n_col8 == 4) && (n_row8 == 4) && (n_col8_2 == 4))
        return tbm_mult_avx2_ncol8_4(in, n_mat, in2, out);
    
    tbm_mult_avx2_256(in, n_mat, n_row8, n_col8, in2, n_col8_2, out);
}
//______________________________________________________________________________
// multiply two 8x8 bit matrices with the second matrix transposed
uint64_t inline tbm_mult_t8x8_avx2(uint64_t a8x8, uint64_t tb8x8)
{
    uint64_t out = 0;
    uint8_t *tb = (uint8_t *)&tb8x8;
    for (uint8_t i_bit = 0; i_bit < 8; i_bit++)
    {
        uint64_t repeat = 0x0101010101010101*tb[7-i_bit];
        uint64_t prod = a8x8 & repeat;
        prod ^= prod << 4;
        prod ^= prod << 2;
        prod ^= prod << 1;
        prod &= 0x8080808080808080;
        out >>= 1;
        out |= prod;
    }
    return out;
}

uint64_t inline tbm_dot_t_avx2(uint64_t a[4], uint64_t b[4], uint32_t n_dot)
{
    uint64_t out = 0;
    for (uint8_t i_dot = 0; i_dot < n_dot; i_dot++)
        out ^= tbm_mult_t8x8_avx2(a[i_dot], b[i_dot]);
    return out;
}

void __attribute__ ((noinline)) tbm_mult_t_avx2_ncol8_1(
    uint64_t *in, uint64_t n_mat, uint64_t *in2t, uint64_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
        out[i_mat] = tbm_mult_t8x8_avx2(in[i_mat], in2t[i_mat]);
}

#pragma GCC push_options //-----------------------------------------------------
#pragma GCC optimize("no-tree-vectorize")
void __attribute__ ((noinline)) tbm_mult_t_dot_avx2(
    uint64_t *in, uint64_t n_mat, uint32_t n_row8, uint32_t n_col8,
    uint64_t *in2, uint32_t n_col8_2, uint64_t *out)
{
    tbm_mult_t_dot_template(
        in, n_mat, n_row8, n_col8, in2, n_col8_2, out, tbm_dot_t_avx2);
}

#pragma GCC pop_options //------------------------------------------------------

void tbm_mult_t_avx2(
    uint64_t *in, uint64_t n_mat, uint32_t n_row8, uint32_t n_col8,
    uint64_t *in2, uint32_t n_row8_2, uint64_t *out)
{
    if ((n_col8 == 1) && (n_row8 == 1) && (n_row8_2 == 1))
        return tbm_mult_t_avx2_ncol8_1(in, n_mat, in2, out);
    // if ((n_col8 == 2) && (n_row8 == 2) && (n_row8_2 == 2))
    //     return tbm_mult_t_avx2_ncol8_2(in, n_mat, in2, out);
    // if ((n_col8 == 4) && (n_row8 == 4) && (n_row8_2 == 4))
    //     return tbm_mult_t_avx2_ncol8_4(in, n_mat, in2, out);
    
    tbm_mult_t_dot_avx2(in, n_mat, n_row8, n_col8, in2, n_row8_2, out);
}
