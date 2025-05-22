#if !defined(_TINYBITMAT_GFNIO)
#define _TINYBITMAT_GFNIO

void tbm_encode_gnfio(
    uint8_t *in, uint64_t n_mat, uint8_t n_bit_col, uint8_t n_bit_row, 
    uint8_t n_octet_col, uint8_t n_octet_row, uint64_t *out);

//______________________________________________________________________________

// typedef void tbm_2arg_int8_fun_t(
//     uint8_t *in, uint8_t *in2, uint64_t n_mat, uint8_t *out);
// typedef void tbm_2arg_int16_fun_t(
//     uint16_t *in, uint16_t *in2, uint64_t n_mat, uint16_t *out);
// typedef void tbm_2arg_int32_fun_t(
//     uint32_t *in, uint32_t *in2, uint64_t n_mat, uint32_t *out);

// uint64_t tbm_transpose8x8_uint64(uint64_t in8x8);
// tbm_1arg_int8_fun_t tbm_transpose8x8;
// tbm_1arg_int16_fun_t tbm_transpose16x16;
// tbm_1arg_int32_fun_t tbm_transpose32x32;

// //______________________________________________________________________________
// uint64_t tbm_mult8x8_uint64(uint64_t a, uint8_t b[8]);
// tbm_2arg_int8_fun_t tbm_mult8x8;
// tbm_2arg_int16_fun_t tbm_mult16x16;
// tbm_2arg_int32_fun_t tbm_mult32x32;

// //______________________________________________________________________________
// uint64_t tbm_mult_t8x8_uint64(uint64_t a8x8, uint8_t tb[8]);
// tbm_2arg_int8_fun_t tbm_mult_t8x8;
// tbm_2arg_int16_fun_t tbm_mult_t16x16;
// tbm_2arg_int32_fun_t tbm_mult_t32x32;
        
#endif
