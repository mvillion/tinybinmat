#if !defined(_TINYBITMAT)
#define _TINYBITMAT
#include "math.h"
#include "stdbool.h"
#include "stdint.h"
#include "stdio.h"

#ifdef _MSC_VER
#define __UNUSED__
#else
#define __UNUSED__ __attribute__((unused))
#endif
void tbm_encode_gfnio(
    uint8_t *in, uint64_t n_mat, uint32_t n_row, uint32_t n_col, uint64_t *out);

void tbm_sprint8_gfnio(
    uint64_t *mat, uint64_t n_mat, uint32_t n_row, uint32_t n_col, char *str01,
    uint8_t *out);

typedef void tbm_transpose_fun_t(
    uint64_t *in, uint64_t n_mat, uint32_t n_row8, uint32_t n_col8, 
    uint64_t *out);

typedef void tbm_mult_fun_t(
    uint64_t *in, uint64_t n_mat, uint32_t n_row8, uint32_t n_col8,
    uint64_t *in2, uint32_t n_col8_2, uint64_t *out);
    
tbm_transpose_fun_t tbm_transpose_u64;
tbm_mult_fun_t tbm_mult_u64;
tbm_mult_fun_t tbm_mult_t_u64;

#endif
