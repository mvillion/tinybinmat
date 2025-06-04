#if !defined(_TINYBITMAT_GFNIO)
#define _TINYBITMAT_GFNIO

void tbm_encode_gfnio(
    uint8_t *in, uint64_t n_mat, uint32_t n_row, uint32_t n_col, uint64_t *out);

void tbm_sprint8_gfnio(
    uint64_t *mat, uint64_t n_mat, uint32_t n_row, uint32_t n_col, char *str01,
    uint8_t *out);
    
//______________________________________________________________________________
void tbm_transpose_gfnio(
    uint64_t *in, uint64_t n_mat, uint32_t n_row8, uint32_t n_col8, 
    uint64_t *out);

// //______________________________________________________________________________
// void tbm_mult_gfnio(
//     uint64_t *in, uint64_t n_mat, uint32_t n_row8, uint32_t n_col8,
//     uint64_t *in2, uint32_t n_col8_2, uint64_t *out);

//______________________________________________________________________________
void tbm_mult_t_gfnio(
    uint64_t *in, uint64_t n_mat, uint32_t n_row8, uint32_t n_col8,
    uint64_t *in2, uint32_t n_row8_2, uint64_t *out);
  
#endif
