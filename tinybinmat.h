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

void tbm_encode8(
    uint8_t *in, uint64_t n_mat, uint8_t n_bit, uint8_t n_bit_raw, 
    uint8_t *out);

void tbm_encode16(
    uint8_t *in, uint64_t n_mat, uint8_t n_bit, uint8_t n_bit_raw, 
    uint16_t *out);

void tbm_encode32(
    uint8_t *in, uint64_t n_mat, uint8_t n_bit, uint8_t n_bit_raw, 
    uint32_t *out);

//______________________________________________________________________________
void tbm_print8(
    uint8_t *mat_list, uint64_t n_mat, uint8_t n_bit, char *str01);

void tbm_print16(
    uint16_t *mat_list, uint64_t n_mat, uint8_t n_bit, char *str01);

void tbm_sprint8(
    uint8_t *mat_list, uint64_t n_mat, uint8_t n_bit, char *str01,
    uint8_t *out);

void tbm_sprint8_avx2(
    uint8_t *mat_list, uint64_t n_mat, char *str01, uint8_t *out);

void tbm_sprint16(
    uint16_t *mat_list, uint64_t n_mat, uint8_t n_bit, char *str01,
    uint8_t *out);

void tbm_sprint32(
    uint32_t *mat_list, uint64_t n_mat, uint8_t n_bit, char *str01,
    uint8_t *out);

//______________________________________________________________________________
void tbm_transpose8x8(uint8_t *in, uint64_t n_mat, uint8_t *out);
void tbm_transpose16x16(uint16_t *in, uint64_t n_mat, uint16_t *out);
void tbm_transpose32x32(uint32_t *in, uint64_t n_mat, uint32_t *out);

//______________________________________________________________________________
void tbm_mult8x8(uint8_t *in, uint8_t *in2, uint64_t n_mat, uint8_t *out);
void tbm_mult16x16(uint16_t *in, uint16_t *in2, uint64_t n_mat, uint16_t *out);

//______________________________________________________________________________
void tbm_mult_t8x8(uint8_t *in, uint8_t *intb, uint64_t n_mat, uint8_t *out);
void tbm_mult_t16x16(
    uint16_t *in, uint16_t *in2t, uint64_t n_mat, uint16_t *out);
void tbm_mult_t32x32(
    uint32_t *in, uint32_t *in2t, uint64_t n_mat, uint32_t *out);
        
#endif
