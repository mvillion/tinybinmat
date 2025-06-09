#if !defined(_TINYBITMAT_AVX2)
#define _TINYBITMAT_AVX2

#if defined(__x86_64__) || defined(_M_X64)
tbm_transpose_fun_t tbm_transpose_avx2;
tbm_mult_fun_t tbm_mult_avx2;
tbm_mult_fun_t tbm_mult_t_avx2;

tbm_transpose_fun_t tbm_transpose_gfni;
tbm_mult_fun_t tbm_mult_gfni;
tbm_mult_fun_t tbm_mult_t_gfni;
#else
#define tbm_transpose_avx2 NULL
#define tbm_mult_avx2 NULL
#define tbm_mult_t_avx2 NULL
#define tbm_transpose_gfni NULL
#define tbm_mult_gfni NULL
#define tbm_mult_t_gfni NULL
#endif

#endif
