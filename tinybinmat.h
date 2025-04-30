#if !defined(_TINYBITMAT)
#define _TINYBITMAT
#include "math.h"
#include "immintrin.h"
#include "stdbool.h"
#include "stdint.h"
#include "stdio.h"

#ifdef _MSC_VER
#define __UNUSED__
#else
#define __UNUSED__ __attribute__((unused))
#endif

extern void tbm_print8(
    uint8_t *mat_list, uint64_t n_mat, uint8_t n_bit, char *str01);

extern void tbm_print16(
    uint16_t *mat_list, uint64_t n_mat, uint8_t n_bit, char *str01);

extern void tbm_sprint8(
    uint8_t *mat_list, uint64_t n_mat, uint8_t n_bit, char *str01,
    uint8_t *out);

extern void tbm_sprint8_avx2(
    uint8_t *mat_list, uint64_t n_mat, char *str01, uint8_t *out);

extern void tbm_sprint16(
    uint16_t *mat_list, uint64_t n_mat, uint8_t n_bit, char *str01,
    uint8_t *out);

extern void tbm_sprint32(
    uint32_t *mat_list, uint64_t n_mat, uint8_t n_bit, char *str01,
    uint8_t *out);

extern void tbm_transpose8x8(
    uint64_t *in8x8, uint64_t n_mat, uint64_t *out8x8);

extern void tbm_transpose16x16(
    uint64_t *in2x16, uint64_t n_mat, uint64_t *out2x16);

extern void tbm_transpose32x32(
    uint64_t *in2x16, uint64_t n_mat, uint64_t *out2x16);

void tbm_mult_t8x8(
    uint64_t *in8x8, uint64_t *tb8x8, uint64_t n_mat, uint64_t *out8x8);

void tbm_mult_t16x16(
    uint64_t *in4x16, uint64_t *tb4x16, uint64_t n_mat, uint64_t *out4x16);
    
#endif
