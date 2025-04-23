#include <tinybinmat_utils.h>

//______________________________________________________________________________
static PyObject* tbm_print(PyObject *self, PyObject *arg)
{
    PyArrayObject *arr_in;
    char *str01; // two characters for 0 and 1
    uint32_t n_bit;

    // if (!PyArg_ParseTuple(arg, "O!I", &PyArray_Type, &arr_in, &n_bit))
    if (!PyArg_ParseTuple(arg, "O!Is", &PyArray_Type, &arr_in, &n_bit, &str01))
        return failure(PyExc_RuntimeError, "failed to parse parameters");
    if (arr_in == NULL) return NULL;

    if (strlen(str01) < 2)
        return failure(PyExc_RuntimeError, "need at least 2 characters");

    // // create output dimensions
    // // last axis is removed, replaced by complex data i&q
    int n_dim = PyArray_NDIM(arr_in);
    if (n_dim < 1)
        return failure(PyExc_RuntimeError, "import need at least 1 dimension");
    npy_intp n_bit_raw = PyArray_DIM(arr_in, n_dim-1);
    npy_intp n_mat = 1;
    for (uint8_t i_dim = 0; i_dim < n_dim-1; i_dim++)
        n_mat *= PyArray_DIM(arr_in, i_dim);

    if ((n_bit < 1) || (n_bit_raw < n_bit))
        return failure(
            PyExc_RuntimeError, 
            "n_bit shall be inferior to the last dimension");

    // ensure the input array is contiguous.
    // PyArray_GETCONTIGUOUS will increase the reference count.
    arr_in = PyArray_GETCONTIGUOUS(arr_in);

    int py_type = PyArray_TYPE(arr_in);
    if (py_type == NPY_UINT16)
    {
        if (n_bit_raw != 8*sizeof(uint16_t))
            return failure(
                PyExc_RuntimeError, "last dimension shall be 16 for uint16");
        uint16_t *data = (uint16_t *)PyArray_DATA(arr_in);
        for (npy_intp i_mat = 0; i_mat < n_mat; i_mat++)
        {
            for (uint8_t i_row = 0; i_row < n_bit; i_row++)
            {
                uint16_t row = data[i_row];
                for (uint8_t i_bit = 0; i_bit < n_bit; i_bit++)
                {
                    printf("%c", str01[row & 1]);
                    row >>= 1;
                }
                printf("\n");
            }
            data += n_bit_raw;
            printf("\n");
        }
    }

    // decrease the reference count
    Py_DECREF(arr_in);
    Py_RETURN_NONE;
}

void inline transpose8x8_int16(
    __m256i hgfedcba4_hgfedcba0, __m256i hgfedcba5_hgfedcba1,
    __m256i hgfedcba6_hgfedcba2, __m256i hgfedcba7_hgfedcba3,
    __m256i *b76543210_a76543210, __m256i *d76543210_c76543210,
    __m256i *f76543210_e76543210, __m256i *h76543210_g76543210)
{
    __m256i d5d4c5c4b5b4a5a4_d1d0c1c0b1b0a1a0 = _mm256_unpacklo_epi16(
        hgfedcba4_hgfedcba0, hgfedcba5_hgfedcba1);
    __m256i h5h4g5g4f5f4e5e4_h1h0g1g0f1f0e1e0 = _mm256_unpackhi_epi16(
        hgfedcba4_hgfedcba0, hgfedcba5_hgfedcba1);
    __m256i d7d6c7c6b7b6a7a6_d3d2c3c2b3b2a3a2 = _mm256_unpacklo_epi16(
        hgfedcba6_hgfedcba2, hgfedcba7_hgfedcba3);
    __m256i h7h6g7g6f7f6e7e6_h3h2g3g2f3f2e3e2 = _mm256_unpackhi_epi16(
        hgfedcba6_hgfedcba2, hgfedcba7_hgfedcba3);

    __m256i b7b6b5b4a7a6a5a4_b3b2b1b0a3a2a1a0 = _mm256_unpacklo_epi32(
        d5d4c5c4b5b4a5a4_d1d0c1c0b1b0a1a0, d7d6c7c6b7b6a7a6_d3d2c3c2b3b2a3a2);
    __m256i d7d6d5d4c7c6c5c4_d3d2d1d0c3c2c1c0 = _mm256_unpackhi_epi32(
        d5d4c5c4b5b4a5a4_d1d0c1c0b1b0a1a0, d7d6c7c6b7b6a7a6_d3d2c3c2b3b2a3a2);
    __m256i f7f6f5f4e7e6e5e4_f3f2f1f0e3e2e1e0 = _mm256_unpacklo_epi32(
        h5h4g5g4f5f4e5e4_h1h0g1g0f1f0e1e0, h7h6g7g6f7f6e7e6_h3h2g3g2f3f2e3e2);
    __m256i h7h6h5h4g7g6g5g4_h3h2h1h0g3g2g1g0 = _mm256_unpackhi_epi32(
        h5h4g5g4f5f4e5e4_h1h0g1g0f1f0e1e0, h7h6g7g6f7f6e7e6_h3h2g3g2f3f2e3e2);

    *b76543210_a76543210 = _mm256_permute4x64_epi64(
        b7b6b5b4a7a6a5a4_b3b2b1b0a3a2a1a0, _MM_SHUFFLE(3, 1, 2, 0));
    *d76543210_c76543210 = _mm256_permute4x64_epi64(
        d7d6d5d4c7c6c5c4_d3d2d1d0c3c2c1c0, _MM_SHUFFLE(3, 1, 2, 0));
    *f76543210_e76543210 = _mm256_permute4x64_epi64(
        f7f6f5f4e7e6e5e4_f3f2f1f0e3e2e1e0, _MM_SHUFFLE(3, 1, 2, 0));
    *h76543210_g76543210 = _mm256_permute4x64_epi64(
        h7h6h5h4g7g6g5g4_h3h2h1h0g3g2g1g0, _MM_SHUFFLE(3, 1, 2, 0));
}

//______________________________________________________________________________
// set up the methods table
static PyMethodDef method_def[] = {
    {"print", tbm_print, METH_VARARGS, "print tinybinmat"},
    {NULL, NULL, 0, NULL}
};

// initialize module
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "tinybinmat",
    NULL,
    -1,
    method_def,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_tinybinmat(void)
{
    // PyObject *d;
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }

    // import_array();
    // import_umath();

    // d = PyModule_GetDict(m);

    import_array();

    return m;
}
