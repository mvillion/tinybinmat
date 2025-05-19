#if !defined(_TINYBITMAT_GFNI)
#define _TINYBITMAT_GFNI
//______________________________________________________________________________
void tbm_transpose8x8_gfni(uint8_t *in, uint64_t n_mat, uint8_t *out);
void tbm_transpose16x16_gfni(uint16_t *in, uint64_t n_mat, uint16_t *out);
void tbm_transpose32x32_gfni(uint32_t *in, uint64_t n_mat, uint32_t *out);

//______________________________________________________________________________
void tbm_mult8x8_gfni(uint8_t *in, uint8_t *in2, uint64_t n_mat, uint8_t *out);
void tbm_mult16x16_gfni(
    uint16_t *in, uint16_t *in2, uint64_t n_mat, uint16_t *out);
void tbm_mult32x32_gfni(
    uint32_t *in, uint32_t *in2, uint64_t n_mat, uint32_t *out);

//______________________________________________________________________________
void tbm_mult_t8x8_gfni(
    uint8_t *in, uint8_t *intb, uint64_t n_mat, uint8_t *out);
void tbm_mult_t16x16_gfni(
    uint16_t *in, uint16_t *in2t, uint64_t n_mat, uint16_t *out);
void tbm_mult_t32x32_gfni(
    uint32_t *in, uint32_t *in2t, uint64_t n_mat, uint32_t *out);
        
#endif
