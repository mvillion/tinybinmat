#include "tinybinmat.h"

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

void tbm_encode8(
    uint8_t *in, uint64_t n_mat, uint8_t n_bit, uint8_t n_bit_raw, 
    uint8_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        for (uint8_t i_bit_raw = 0; i_bit_raw < n_bit; i_bit_raw++)
        {
            uint8_t acc = 0;
            for (uint8_t i_bit = 0; i_bit < n_bit; i_bit++)
            {
                uint8_t bit = in[(i_mat*n_bit+i_bit_raw)*n_bit+i_bit];
                acc |= (bit & 1) << i_bit;
            }
            out[i_mat*n_bit_raw+i_bit_raw] = acc;
        }
        for (uint8_t i_bit_raw = n_bit; i_bit_raw < n_bit_raw; i_bit_raw++)
            out[i_mat*n_bit_raw+i_bit_raw] = 0;
    }
}

void tbm_encode16(
    uint8_t *in, uint64_t n_mat, uint8_t n_bit, uint8_t n_bit_raw, 
    uint16_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        for (uint8_t i_bit_raw = 0; i_bit_raw < n_bit; i_bit_raw++)
        {
            uint16_t acc = 0;
            for (uint8_t i_bit = 0; i_bit < n_bit; i_bit++)
            {
                uint8_t bit = in[(i_mat*n_bit+i_bit_raw)*n_bit+i_bit];
                acc |= (bit & 1) << i_bit;
            }
            out[i_mat*n_bit_raw+i_bit_raw] = acc;
        }
        for (uint8_t i_bit_raw = n_bit; i_bit_raw < n_bit_raw; i_bit_raw++)
            out[i_mat*n_bit_raw+i_bit_raw] = 0;
    }
}

void tbm_encode32(
    uint8_t *in, uint64_t n_mat, uint8_t n_bit, uint8_t n_bit_raw, 
    uint32_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        for (uint8_t i_bit_raw = 0; i_bit_raw < n_bit; i_bit_raw++)
        {
            uint32_t acc = 0;
            for (uint8_t i_bit = 0; i_bit < n_bit; i_bit++)
            {
                uint8_t bit = in[(i_mat*n_bit+i_bit_raw)*n_bit+i_bit];
                acc |= (bit & 1) << i_bit;
            }
            out[i_mat*n_bit_raw+i_bit_raw] = acc;
        }
        for (uint8_t i_bit_raw = n_bit; i_bit_raw < n_bit_raw; i_bit_raw++)
            out[i_mat*n_bit_raw+i_bit_raw] = 0;
    }
}

void tbm_print8(
    uint8_t *mat_list, uint64_t n_mat, uint8_t n_bit, char *str01)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        for (uint8_t i_row = 0; i_row < n_bit; i_row++)
        {
            uint8_t row = mat_list[i_row];
            for (uint8_t i_bit = 0; i_bit < n_bit; i_bit++)
            {
                printf("%c", str01[row & 1]);
                row >>= 1;
            }
            printf("\n");
        }
        mat_list += 8*sizeof(uint8_t);
        printf("\n");
    }
}

void tbm_print16(
    uint16_t *mat_list, uint64_t n_mat, uint8_t n_bit, char *str01)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        for (uint8_t i_row = 0; i_row < n_bit; i_row++)
        {
            uint16_t row = mat_list[i_row];
            for (uint8_t i_bit = 0; i_bit < n_bit; i_bit++)
            {
                printf("%c", str01[row & 1]);
                row >>= 1;
            }
            printf("\n");
        }
        mat_list += 8*sizeof(uint16_t);
        printf("\n");
    }
}

void tbm_sprint8(
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

void tbm_sprint8_avx2(
    uint8_t *mat_list, uint64_t n_mat, char *str01, uint8_t *out)
{
    uint64_t *mat_list64 = (uint64_t *)mat_list;
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        __m256i mask32 = _mm256_movm_epi8_avx2(mat_list64[i_mat] & 0xffffffff);
        _mm256_storeu_si256((__m256i *)out, mask32);
        out += 8*4;
        mask32 = _mm256_movm_epi8_avx2(mat_list64[i_mat] >> 32);
        _mm256_storeu_si256((__m256i *)out, mask32);
        out += 8*4;
    }
}

void tbm_sprint16(
    uint16_t *mat_list, uint64_t n_mat, uint8_t n_bit, char *str01,
    uint8_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        for (uint8_t i_row = 0; i_row < n_bit; i_row++)
        {
            uint16_t row = mat_list[i_row];
            for (uint8_t i_bit = 0; i_bit < n_bit; i_bit++)
            {
                *out++ = str01[row & 1];
                row >>= 1;
            }
        }
        mat_list += 8*sizeof(uint16_t);
    }
}

void tbm_sprint32(
    uint32_t *mat_list, uint64_t n_mat, uint8_t n_bit, char *str01,
    uint8_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        for (uint8_t i_row = 0; i_row < n_bit; i_row++)
        {
            uint32_t row = mat_list[i_row];
            for (uint8_t i_bit = 0; i_bit < n_bit; i_bit++)
            {
                *out++ = str01[row & 1];
                row >>= 1;
            }
        }
        mat_list += 8*sizeof(uint32_t);
    }
}

//______________________________________________________________________________
uint64_t inline tbm_transpose8x8_uint64(uint64_t in8x8)
{
    // input is 8x8 bit matrix with 8 rows: 0x0706050403020100
    // 1st bit in rows is LSB, thus reversed compared to matrix notation
    uint64_t ur_mask4x4 = 0x00000000f0f0f0f0; // up right 4x4 bits
    // uint64_t = dl_mask4x4 = 0x0f0f0f0f00000000; // down left 4x4 bits
    // dl_mask4x4 == ur_mask4x4 << (4*8-4)
    uint64_t xor = in8x8 ^ (in8x8 >> 28);
    xor &= ur_mask4x4;
    in8x8 ^= xor;
    xor <<= 28;
    in8x8 ^= xor;
    uint64_t ur_mask2x2 = 0x0000cccc0000cccc; // 4 up right 2x2 bits
    // uint64_t = dl_mask2x2 = 0x3333000033330000; // 4 down left 2x2 bits
    // dl_mask2x2 == ur_mask2x2 << (2*8-2)
    xor = in8x8 ^ (in8x8 >> 14);
    xor &= ur_mask2x2;
    in8x8 ^= xor;
    xor <<= 14;
    in8x8 ^= xor;
    uint64_t ur_mask1x1 = 0x00aa00aa00aa00aa; // 16 up right 1x1 bits
    // uint64_t = dl_mask1x1 = 0x5500550055005500; // 16 down left 1x1 bits
    // dl_mask1x1 == ur_mask1x1 << (8-1)
    xor = in8x8 ^ (in8x8 >> 7);
    xor &= ur_mask1x1;
    in8x8 ^= xor;
    xor <<= 7;
    in8x8 ^= xor;
    return in8x8;
}

void inline tbm_transpose16x16_uint64(
    uint64_t in03x16, uint64_t in47x16, uint64_t in8bx16, uint64_t incfx16,
    uint64_t *out03x16, uint64_t *out47x16, uint64_t *out8bx16, 
    uint64_t *outcfx16)
{
    // inputs are 4x16 bit matrix with 4 rows: 0x00030002_00010000
    // 1st bit in rows is LSB, thus reversed compared to matrix notation
    uint64_t ur_mask4x8 = 0xff00ff00ff00ff00; // up right 4x8 bits
    // uint64_t = dl_mask4x8 = 0x00ff00ff00ff00ff; // down left 4x8 bits
    // dl_mask4x8 == ur_mask4x8 >> 8
    uint64_t xor0 = in03x16 ^ (in8bx16 << 8);
    uint64_t xor1 = in47x16 ^ (incfx16 << 8);
    xor0 &= ur_mask4x8;
    xor1 &= ur_mask4x8;
    in03x16 ^= xor0;
    in47x16 ^= xor1;
    xor0 >>= 8;
    xor1 >>= 8;
    in8bx16 ^= xor0;
    incfx16 ^= xor1;
    uint64_t ur_mask4x4 = 0xf0f0f0f0f0f0f0f0; // 2 up right 4x4 bits
     // uint64_t = dl_mask4x4 = 0x0f0f0f0f0f0f0f0f; // 2 down left 4x4 bits
    // dl_mask4x4 == ur_mask4x4 >> 4
    xor0 = in03x16 ^ (in47x16 << 4);
    xor1 = in8bx16 ^ (incfx16 << 4);
    xor0 &= ur_mask4x4;
    xor1 &= ur_mask4x4;
    in03x16 ^= xor0;
    in8bx16 ^= xor1;
    xor0 >>= 4;
    xor1 >>= 4;
    in47x16 ^= xor0;
    incfx16 ^= xor1;
    uint64_t ur_mask2x2 = 0x00000000cccccccc; // 4 up right 2x2 bits
    // uint64_t = dl_mask2x2 = 0x3333333300000000; // 4 down left 2x2 bits
    // dl_mask2x2 == ur_mask2x2 << (16*2-2)
    xor0 = in03x16 ^ (in03x16 >> 30);
    xor0 &= ur_mask2x2; in03x16 ^= xor0; xor0 <<= 30; in03x16 ^= xor0;
    xor0 = in47x16 ^ (in47x16 >> 30);
    xor0 &= ur_mask2x2; in47x16 ^= xor0; xor0 <<= 30; in47x16 ^= xor0;
    xor0 = in8bx16 ^ (in8bx16 >> 30);
    xor0 &= ur_mask2x2; in8bx16 ^= xor0; xor0 <<= 30; in8bx16 ^= xor0;
    xor0 = incfx16 ^ (incfx16 >> 30);
    xor0 &= ur_mask2x2; incfx16 ^= xor0; xor0 <<= 30; incfx16 ^= xor0;
    uint64_t ur_mask1x1 = 0x0000aaaa0000aaaa; // 16 up right 1x1 bits
    // uint64_t = dl_mask1x1 = 0x5555000055550000; // 16 down left 1x1 bits
    // dl_mask1x1 == ur_mask1x1 << (16-1)
    xor0 = in03x16 ^ (in03x16 >> 15);
    xor0 &= ur_mask1x1; in03x16 ^= xor0; xor0 <<= 15; in03x16 ^= xor0;
    xor0 = in47x16 ^ (in47x16 >> 15);
    xor0 &= ur_mask1x1; in47x16 ^= xor0; xor0 <<= 15; in47x16 ^= xor0;
    xor0 = in8bx16 ^ (in8bx16 >> 15);
    xor0 &= ur_mask1x1; in8bx16 ^= xor0; xor0 <<= 15; in8bx16 ^= xor0;
    xor0 = incfx16 ^ (incfx16 >> 15);
    xor0 &= ur_mask1x1; incfx16 ^= xor0; xor0 <<= 15; incfx16 ^= xor0;
    *out03x16 = in03x16;
    *out47x16 = in47x16;
    *out8bx16 = in8bx16;
    *outcfx16 = incfx16;
}

void inline tbm_transpose32x32_uint64(uint64_t in_read[16], uint64_t in[16])
{
    for (uint8_t i_row = 0; i_row < 16; i_row++)
        in[i_row] = in_read[i_row];
    // inputs are 2x32 bit matrix with 2 rows: 0x00000001_00000000
    // 1st bit in rows is LSB, thus reversed compared to matrix notation
    uint64_t ur_mask2x16 = 0xffff0000ffff0000; // up right 2x16 bits
    // uint64_t = dl_mask2x16 = 0x0000ffff0000ffff; // down left 2x16 bits
    // dl_mask2x16 == ur_mask2x16 >> 16
    uint64_t xor;
    for (uint8_t i_row = 0; i_row < 8; i_row++)
    {
        xor = in[i_row] ^ (in[i_row+8] << 16);
        xor &= ur_mask2x16;
        in[i_row] ^= xor;
        xor >>= 16;
        in[i_row+8] ^= xor;
    }
    uint64_t ur_mask2x8 = 0xff00ff00ff00ff00; // 2 up right 2x8 bits
    // uint64_t = dl_mask2x8 = 0x00ff00ff00ff00ff; // 2 down left 2x16 bits
    // dl_mask2x8 == ur_mask2x8 >> 8
    for (uint8_t i_block = 0; i_block < 2; i_block++)
        for (uint8_t i_row = 0; i_row < 4; i_row++)
        {
            xor = in[i_block*8+i_row] ^ (in[i_block*8+i_row+4] << 8);
            xor &= ur_mask2x8;
            in[i_block*8+i_row] ^= xor;
            xor >>= 8;
            in[i_block*8+i_row+4] ^= xor;
        }
    uint64_t ur_mask2x4 = 0xf0f0f0f0f0f0f0f0; // 4 up right 2x4 bits
    // uint64_t = dl_mask2x4 = 0x0f0f0f0f0f0f0f0f; // 4 down left 2x4 bits
    // dl_mask2x4 == ur_mask2x4 >> 4
    for (uint8_t i_block = 0; i_block < 4; i_block++)
        for (uint8_t i_row = 0; i_row < 2; i_row++)
        {
            xor = in[i_block*4+i_row] ^ (in[i_block*4+i_row+2] << 4);
            xor &= ur_mask2x4;
            in[i_block*4+i_row] ^= xor;
            xor >>= 4;
            in[i_block*4+i_row+2] ^= xor;
        }
    uint64_t ur_mask2x2 = 0xcccccccccccccccc; // 8 up right 2x2 bits
    // uint64_t = dl_mask2x2 = 0x3333333333333333; // 4 down left 2x2 bits
    // dl_mask2x2 == ur_mask2x2 >> 2
    for (uint8_t i_block = 0; i_block < 8; i_block++)
        for (uint8_t i_row = 0; i_row < 1; i_row++)
        {
            xor = in[i_block*2+i_row] ^ (in[i_block*2+i_row+1] << 2);
            xor &= ur_mask2x2;
            in[i_block*2+i_row] ^= xor;
            xor >>= 2;
            in[i_block*2+i_row+1] ^= xor;
        }
    uint64_t ur_mask1x1 = 0x00000000aaaaaaaa; // 16 up right 1x1 bits
    // uint64_t = dl_mask1x1 = 0x5555555500000000; // 16 down left 1x1 bits
    // dl_mask1x1 == ur_mask1x1 << (32-1)
    for (uint8_t i_row = 0; i_row < 16; i_row++)
    {
        xor = in[i_row] ^ (in[i_row] >> 31);
        xor &= ur_mask1x1;
        in[i_row] ^= xor; xor <<= 31; in[i_row] ^= xor;
    }
    return;
}

void tbm_transpose8x8(uint8_t *in, uint64_t n_mat, uint8_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        *((uint64_t *)out) = tbm_transpose8x8_uint64(*((uint64_t *)in));
        in += 8;
        out += 8;
    }
}

void tbm_transpose16x16(uint16_t *in, uint64_t n_mat, uint16_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        uint64_t *in4x16 = (uint64_t *)in;
        uint64_t *out4x16 = (uint64_t *)out;
        tbm_transpose16x16_uint64(
            in4x16[0], in4x16[1], in4x16[2], in4x16[3],
            out4x16+0, out4x16+1, out4x16+2, out4x16+3);
        in += 16;
        out += 16;
    }
}

void tbm_transpose32x32(uint32_t *in, uint64_t n_mat, uint32_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        tbm_transpose32x32_uint64((uint64_t *)in, (uint64_t *)out);
        in += 32;
        out += 32;
    }
}

//______________________________________________________________________________
// multiply two 8x8 bit matrices
uint64_t inline tbm_mult8x8_uint64(uint64_t a, uint8_t b[8])
{
    uint64_t out = 0;
    for (uint8_t i_bit = 0; i_bit < 8; i_bit++)
    {
        // create bit mask from the least significant bits in a
        uint64_t bit_a = a & 0x0101010101010101;
        bit_a *= 0xff;
        a >>= 1;
        uint64_t prod = bit_a & (0x0101010101010101*b[i_bit]);
        out ^= prod;
    }
    return out;
}

// multiply two 16x16 bit matrices
void inline tbm_mult16x16_uint64(uint64_t a[4], uint16_t b[16], uint64_t out[4])
{
    uint64_t a_bck[4];
    for (uint8_t i_4row = 0; i_4row < 4; i_4row++)
    {
        a_bck[i_4row] = a[i_4row];
    }
    for (uint8_t i_4row = 0; i_4row < 4; i_4row++)
    {
        out[i_4row] = 0;
        for (uint8_t i_bit = 0; i_bit < 16; i_bit++)
        {
            // create bit mask from the least significant bits in a
            uint64_t bit_a = a_bck[i_4row] & 0x0001000100010001;
            bit_a *= 0xffff;
            a_bck[i_4row] >>= 1;
            uint64_t prod = bit_a & (0x0001000100010001*b[i_bit]);
            out[i_4row] ^= prod;
        }
    }
}

// multiply two 32x32 bit matrices
void inline tbm_mult32x32_uint64(
    uint64_t a[16], uint32_t b[32], uint64_t out[16])
{
    uint64_t a_bck[16];
    for (uint8_t i_2row = 0; i_2row < 16; i_2row++)
    {
        a_bck[i_2row] = a[i_2row];
    }
    for (uint8_t i_2row = 0; i_2row < 16; i_2row++)
    {
        out[i_2row] = 0;
        for (uint8_t i_bit = 0; i_bit < 32; i_bit++)
        {
            // create bit mask from the least significant bits in a
            uint64_t bit_a = a_bck[i_2row] & 0x0000000100000001;
            bit_a *= 0xffffffff;
            a_bck[i_2row] >>= 1;
            uint64_t prod = bit_a & (0x0000000100000001*b[i_bit]);
            out[i_2row] ^= prod;
        }
    }
}

void tbm_mult8x8(uint8_t *in, uint8_t *in2, uint64_t n_mat, uint8_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        *((uint64_t *)out) = tbm_mult8x8_uint64(*((uint64_t *)in), in2);
        in += 8;
        in2 += 8;
        out += 8;
    }
}

void tbm_mult16x16(uint16_t *in, uint16_t *in2, uint64_t n_mat, uint16_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        tbm_mult16x16_uint64((uint64_t *)in, in2, (uint64_t *)out);
        in += 16;
        in2 += 16;
        out += 16;
    }
}

void tbm_mult32x32(uint32_t *in, uint32_t *in2, uint64_t n_mat, uint32_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        tbm_mult32x32_uint64((uint64_t *)in, in2, (uint64_t *)out);
        in += 32;
        in2 += 32;
        out += 32;
    }
}

//______________________________________________________________________________
// multiply two 8x8 bit matrices with the second matrix transposed
uint64_t inline tbm_mult_t8x8_uint64(uint64_t a8x8, uint8_t tb[8])
{
    uint64_t out = 0;
    for (uint8_t i_bit = 0; i_bit < 8; i_bit++)
    {
        uint64_t repeat = 0x0101010101010101*tb[i_bit];
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

// multiply two 16x16 bit matrices with the second matrix transposed
// note: this code output is transposed, thus input were swapped...
void inline tbm_mult_t16x16_uint64(
    uint64_t tb4x16[4], uint16_t a[16], uint64_t out4x16[4])
{
    for (uint8_t i_4col = 0; i_4col < 4; i_4col++)
    { 
        uint64_t out = 0;
        uint64_t tb_4col = tb4x16[i_4col];
        uint64_t prod[4];
        for (uint8_t i_bit = 0; i_bit < 4; i_bit++)
        {
            uint64_t repeat = 0x0001000100010001*a[4*0+i_bit];
            prod[0] = tb_4col & repeat;
            prod[0] ^= prod[0] >> 8;
            prod[0] &= 0x00ff00ff00ff00ff;
            repeat = 0x0001000100010001*a[4*2+i_bit];
            prod[2] = tb_4col & repeat;
            prod[2] ^= prod[2] >> 8;
            prod[2] &= 0x00ff00ff00ff00ff;
            prod[0] = prod[0] ^ (prod[2] << 8);
            prod[0] ^= prod[0] >> 4;
            prod[0] &= 0x0f0f0f0f0f0f0f0f;

            repeat = 0x0001000100010001*a[4*1+i_bit];
            prod[1] = tb_4col & repeat;
            prod[1] ^= prod[1] >> 8;
            prod[1] &= 0x00ff00ff00ff00ff;
            repeat = 0x0001000100010001*a[4*3+i_bit];
            prod[3] = tb_4col & repeat;
            prod[3] ^= prod[3] >> 8;
            prod[3] &= 0x00ff00ff00ff00ff;
            prod[1] = prod[1] ^ (prod[3] << 8);
            prod[1] ^= prod[1] >> 4;
            prod[1] &= 0x0f0f0f0f0f0f0f0f;

            prod[0] = prod[0] ^ (prod[1] << 4);
            prod[0] ^= prod[0] << 2;
            prod[0] ^= prod[0] << 1;
            prod[0] &= 0x8888888888888888;
            
            out >>= 1;
            out |= prod[0];
        }
        out4x16[i_4col] = out;
    }
}

// multiply two 32x32 bit matrices with the second matrix transposed
// note: this code output is transposed, thus input were swapped...
void inline tbm_mult_t32x32_uint64(
    uint64_t tb2x32[16], uint32_t a1x32[32], uint64_t out2x32[16])
{
    for (uint8_t i_2col = 0; i_2col < 16; i_2col++)
    { 
        uint64_t out = 0;
        uint64_t tb_2col = tb2x32[i_2col];
        uint64_t prod[8];
        for (uint8_t i_bit = 0; i_bit < 2; i_bit++)
        {
            for (uint8_t i_row = 0; i_row < 8; i_row++)
            {
                uint64_t repeat = 0x0000000100000001*a1x32[2*i_row+i_bit];
                uint64_t prodl = tb_2col & repeat;
                prodl ^= prodl >> 16;
                prodl &= 0x0000ffff0000ffff;
                
                repeat = 0x0000000100000001*a1x32[16+2*i_row+i_bit];
                uint64_t prodh = tb_2col & repeat;
                prodh ^= prodh << 16;
                prodh &= 0xffff0000ffff0000;
                
                prod[i_row] = prodl ^ prodh;
            }
            for (uint8_t i_row = 0; i_row < 4; i_row++)
            {
                uint64_t prodl = prod[i_row];
                prodl ^= prodl >> 8;
                prodl &= 0x00ff00ff00ff00ff;
                uint64_t prodh = prod[i_row+4];
                prodh ^= prodh << 8;
                prodh &= 0xff00ff00ff00ff00;
                prod[i_row] = prodl ^ prodh;
            }
            for (uint8_t i_row = 0; i_row < 2; i_row++)
            {
                uint64_t prodl = prod[i_row];
                prodl ^= prodl >> 4;
                prodl &= 0x0f0f0f0f0f0f0f0f;
                uint64_t prodh = prod[i_row+2];
                prodh ^= prodh << 4;
                prodh &= 0xf0f0f0f0f0f0f0f0;
                prod[i_row] = prodl ^ prodh;
            }
            for (uint8_t i_row = 0; i_row < 1; i_row++)
            {
                uint64_t prodl = prod[i_row];
                prodl ^= prodl >> 2;
                prodl &= 0x3333333333333333;
                uint64_t prodh = prod[i_row+1];
                prodh ^= prodh << 2;
                prodh &= 0xcccccccccccccccc;
                prod[i_row] = prodl ^ prodh;
            }

            prod[0] ^= prod[0] << 1;
            prod[0] &= 0xaaaaaaaaaaaaaaaa;
            
            out >>= 1;
            out |= prod[0];
        }
        out2x32[i_2col] = out;
    }
}

void tbm_mult_t8x8(uint8_t *in, uint8_t *in2t, uint64_t n_mat, uint8_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        *((uint64_t *)out) = tbm_mult_t8x8_uint64(*((uint64_t *)in), in2t);
        in += 8;
        in2t += 8;
        out += 8;
    }
}

void tbm_mult_t16x16(
    uint16_t *in, uint16_t *in2t, uint64_t n_mat, uint16_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        tbm_mult_t16x16_uint64((uint64_t *)in, in2t, (uint64_t *)out);
        in += 16;
        in2t += 16;
        out += 16;
    }
}

void tbm_mult_t32x32(
    uint32_t *in, uint32_t *in2t, uint64_t n_mat, uint32_t *out)
{
    for (uint64_t i_mat = 0; i_mat < n_mat; i_mat++)
    {
        tbm_mult_t32x32_uint64((uint64_t *)in, in2t, (uint64_t *)out);
        in += 32;
        in2t += 32;
        out += 32;
    }
}
