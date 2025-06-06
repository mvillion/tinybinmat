#include "tinybinmat.h"
#include "tinybinmat_avx2.h"
#include "tinybinmat_gfni.h"
#include "tinybinmat_gfnio.h"
#include "tinybinmat_utils.h"


uint8_t parse_method_str(const char *method_str, bool *use_gfnio)
{
    uint8_t i_fun = 0xff;
    *use_gfnio = false;
    if ((method_str == NULL) || (strcmp(method_str, "default") == 0))
    {
        i_fun = 0;
    }
    else if (strcmp(method_str, "avx2") == 0)
    {
        i_fun = 1;
    }
    else if (strcmp(method_str, "gfni") == 0)
    {
        i_fun = 2;
    }
    else if (strcmp(method_str, "gfnio") == 0)
    {
        i_fun = 2;
        *use_gfnio = true;
    }
    return i_fun;    
}

//______________________________________________________________________________
static PyObject* tbm_encode(PyObject *self, PyObject *arg, PyObject *kwarg)
{
    PyArrayObject *arr_in; //!< 1st array of matrices to multiply
    char *format_str = NULL;

    static char *kwlist[] = {"in", "fmt", NULL};
    int ok = PyArg_ParseTupleAndKeywords(
        arg, kwarg, "O!|s", kwlist, &PyArray_Type, &arr_in, &format_str);
    if (!ok)
        return failure(PyExc_RuntimeError, "failed to parse parameters");
    if (arr_in == NULL) return NULL;

    bool use_gfnio;
    if ((format_str == NULL) || (strcmp(format_str, "default") == 0))
    {
        use_gfnio = false;
    }
    else if (strcmp(format_str, "gfni") == 0)
    {
        use_gfnio = true;
    }
    else
        return failure(
            PyExc_RuntimeError, "format shall be 'gfni' or 'default'");

    int n_dim = PyArray_NDIM(arr_in);
    if (n_dim < 2)
        return failure(PyExc_RuntimeError, "input need at least 2 dimension");
    npy_intp n_col = PyArray_DIM(arr_in, n_dim-1);
    npy_intp n_row = PyArray_DIM(arr_in, n_dim-2);

    if (n_col != n_row && !use_gfnio)
        return failure(
            PyExc_RuntimeError, "last two dimensions shall be equal");

    npy_intp size_type = PyArray_ITEMSIZE(arr_in);
    if (size_type != 1)
        return failure(
            PyExc_RuntimeError, "input type size shall be equal to 1 octet");

    uint64_t n_mat = (uint64_t)(PyArray_SIZE(arr_in)/n_col/n_row);

    int py_type;
    uint8_t n_bit_raw;
    int n_dim_out;

    if (use_gfnio)
    {
        py_type = NPY_UINT64;
        n_dim_out = n_dim;
    }
    else
    {
        if ((n_col < 1) || (32 < n_col))
            return failure(
                PyExc_RuntimeError,
                "last two dimensions shall be between 1 and 32 bits");

        if (n_col <= 8)
        {
            py_type = NPY_UINT8; n_bit_raw = 8;
        }
        else if (n_col <= 16)
        {
            py_type = NPY_UINT16; n_bit_raw = 16;
        }
        else if (n_col <= 32)
        {
            py_type = NPY_UINT32; n_bit_raw = 32;
        }
        else
            return failure(
                PyExc_RuntimeError,
                "input type size shall be equal to 1, 2, or 4 octets");
        n_dim_out = n_dim-1;
    }

    // create output dimensions
    npy_intp *out_dim = (npy_intp *)malloc(n_dim_out*sizeof(npy_intp));
    memcpy(out_dim, PyArray_DIMS(arr_in), (n_dim_out-1)*sizeof(npy_intp));
    if (use_gfnio)
    {
        out_dim[n_dim_out-1] = (n_col+7)/8;
        out_dim[n_dim_out-2] = (n_row+7)/8;
    }
    else
        out_dim[n_dim-2] = n_bit_raw;
    PyObject *arr_out = PyArray_SimpleNew(n_dim_out, out_dim, py_type);

    // ensure the input array is contiguous.
    // PyArray_GETCONTIGUOUS will increase the reference count.
    arr_in = PyArray_GETCONTIGUOUS(arr_in);

    uint8_t *in = (uint8_t *)PyArray_DATA(arr_in);
    if (use_gfnio)
    {
        tbm_encode_gfnio(
            in, n_mat, n_row, n_col, 
            (uint64_t *)PyArray_DATA((PyArrayObject *)arr_out));
    }
    else
    {
        if (py_type == NPY_UINT8)
        {
            tbm_encode8(
                in, n_mat, n_col, n_bit_raw,
                (uint8_t *)PyArray_DATA((PyArrayObject *)arr_out));
        }
        else if (py_type == NPY_UINT16)
        {
            tbm_encode16(
                in, n_mat, n_col, n_bit_raw,
                (uint16_t *)PyArray_DATA((PyArrayObject *)arr_out));
        }
        else if (py_type == NPY_UINT32)
        {
            tbm_encode32(
                in, n_mat, n_col, n_bit_raw,
                (uint32_t *)PyArray_DATA((PyArrayObject *)arr_out));
        }
    }

    // decrease the reference count
    Py_DECREF(arr_in);
    free(out_dim);
    return arr_out;
}

static PyObject* tbm_mult_template(
    PyObject *self, PyObject *arg, PyObject *kwarg, bool is_transposed)
{
    PyArrayObject *arr_in; //!< 1st array of matrices to multiply
    PyArrayObject *arr_in2; //!< 2nd array of matrices to multiply (transposed)
    char *method_str = NULL;

    static char *kwlist[] = {"in", "in2", "method", NULL};
    int ok = PyArg_ParseTupleAndKeywords(
        arg, kwarg, "O!O!|s", kwlist, &PyArray_Type, &arr_in, 
        &PyArray_Type, &arr_in2, &method_str);
    if (!ok)
        return PyErr_Format(PyExc_RuntimeError, "failed to parse parameters");
    if (arr_in == NULL) return NULL;

    bool use_gfnio;
    uint8_t i_fun = parse_method_str(method_str, &use_gfnio);
    if (i_fun == 0xff)
        return PyErr_Format(
            PyExc_RuntimeError,
            "method string shall be 'avx2', 'gfni', 'gfnio', or 'default'");
    i_fun += is_transposed ? 3 : 0;

    // create output dimensions
    int n_dim = PyArray_NDIM(arr_in);
    int n_dim2 = PyArray_NDIM(arr_in);
    npy_intp n_mat = PyArray_SIZE(arr_in);
    npy_intp n_mat2 = PyArray_SIZE(arr_in2);
    int n_dim_out = n_dim;
    npy_intp n_col8 = 1;
    npy_intp n_row8 = 1;
    npy_intp n_col8_2 = 1;
    npy_intp n_row8_2 = 1;

    if (use_gfnio)
    {
        if ((n_dim < 2) || (n_dim2 < 2))
            return PyErr_Format(
                PyExc_RuntimeError, "input need at least 2 dimensions");
        n_col8 = PyArray_DIM(arr_in, n_dim-1);
        n_row8 = PyArray_DIM(arr_in, n_dim-2);
        n_col8_2 = PyArray_DIM(arr_in2, n_dim-1);
        n_row8_2 = PyArray_DIM(arr_in2, n_dim-2);
        n_mat /= n_col8*n_row8;
        n_mat2 /= n_col8_2*n_row8_2;

        if (is_transposed && (n_col8 != n_col8_2))
        {
            return PyErr_Format(
                PyExc_RuntimeError,
                "first matrix number of columns (%d) shall be equal to "
                "the second matrix number of columns (%d) when transposed",
                n_col8, n_col8_2);
        }
        else if (!is_transposed && (n_col8 != n_row8_2))
        {
            return PyErr_Format(
                PyExc_RuntimeError,
                "first matrix number of columns (%d) shall be equal to "
                "the second matrix number of rows (%d)", n_col8, n_row8_2);
        }
    }
    else
    {
        if (n_dim < 1)
            return PyErr_Format(
                PyExc_RuntimeError, "input need at least 1 dimension");
        npy_intp n_bit_raw = PyArray_DIM(arr_in, n_dim-1);
    
        npy_intp size_type = PyArray_ITEMSIZE(arr_in);
        if (n_bit_raw != 8*size_type)
            return PyErr_Format(
                PyExc_RuntimeError,
                "last dimension %d shall be the number of bits of the type %d",
                n_bit_raw, 8*size_type);
    
        n_mat /= n_bit_raw;
        n_mat2 /= n_bit_raw;
    }
    if (n_mat != n_mat2)
    return PyErr_Format(
        PyExc_RuntimeError, "two arrays shall have the same size");

    npy_intp *out_dim = (npy_intp *)malloc(n_dim_out*sizeof(npy_intp));
    memcpy(out_dim, PyArray_DIMS(arr_in), n_dim*sizeof(npy_intp));
    if (use_gfnio)
    {
        out_dim[n_dim_out-1] = is_transposed ? n_row8_2 : n_col8_2;
        out_dim[n_dim_out-2] = n_row8;
    }
    int py_type = PyArray_TYPE(arr_in);
    PyObject *arr_out = PyArray_SimpleNew(n_dim_out, out_dim, py_type);

    // ensure the input array is contiguous.
    // PyArray_GETCONTIGUOUS will increase the reference count.
    arr_in = PyArray_GETCONTIGUOUS(arr_in);
    arr_in2 = PyArray_GETCONTIGUOUS(arr_in2);
    free(out_dim);

    if (use_gfnio && (py_type == NPY_UINT64))
    {
        uint64_t *in = (uint64_t *)PyArray_DATA(arr_in);
        uint64_t *in2 = (uint64_t *)PyArray_DATA(arr_in2);
        uint64_t *out = (uint64_t *)PyArray_DATA((PyArrayObject *)arr_out);
        if (is_transposed)
            tbm_mult_t_gfnio(in, n_mat, n_row8, n_col8, in2, n_row8_2, out);
        else
            tbm_mult_gfnio(in, n_mat, n_row8, n_col8, in2, n_col8_2, out);
    }
    else if ((py_type == NPY_INT8) || (py_type == NPY_UINT8))
    {
        uint8_t *in = (uint8_t *)PyArray_DATA(arr_in);
        uint8_t *in2 = (uint8_t *)PyArray_DATA(arr_in2);
        uint8_t *out = (uint8_t *)PyArray_DATA((PyArrayObject *)arr_out);
        tbm_2arg_int8_fun_t *fun[6] = {
            tbm_mult8x8, tbm_mult8x8_avx2, tbm_mult8x8_gfni, 
            tbm_mult_t8x8, tbm_mult_t8x8_avx2, tbm_mult_t8x8_gfni};
        fun[i_fun](in, in2, n_mat, out);
    }
    else if ((py_type == NPY_INT16) || (py_type == NPY_UINT16))
    {
        uint16_t *in = (uint16_t *)PyArray_DATA(arr_in);
        uint16_t *in2 = (uint16_t *)PyArray_DATA(arr_in2);
        uint16_t *out = (uint16_t *)PyArray_DATA((PyArrayObject *)arr_out);
        tbm_2arg_int16_fun_t *fun[6] = {
            tbm_mult16x16, tbm_mult16x16_avx2, tbm_mult16x16_gfni, 
            tbm_mult_t16x16, tbm_mult_t16x16_avx2, tbm_mult_t16x16_gfni};
        fun[i_fun](in, in2, n_mat, out);
    }
    else if ((py_type == NPY_INT32) || (py_type == NPY_UINT32))
    {
        uint32_t *in = (uint32_t *)PyArray_DATA(arr_in);
        uint32_t *in2 = (uint32_t *)PyArray_DATA(arr_in2);
        uint32_t *out = (uint32_t *)PyArray_DATA((PyArrayObject *)arr_out);
        tbm_2arg_int32_fun_t *fun[6] = {
            tbm_mult32x32, tbm_mult32x32_avx2, tbm_mult32x32_gfni, 
            tbm_mult_t32x32, tbm_mult_t32x32_avx2, tbm_mult_t32x32_gfni};
        fun[i_fun](in, in2, n_mat, out);
    }
    else
        failure(PyExc_RuntimeError, "input type is not supported");

    // decrease the reference count
    Py_DECREF(arr_in2);
    Py_DECREF(arr_in);
    return arr_out;
}

static PyObject* tbm_mult(PyObject *self, PyObject *arg, PyObject *kwarg)
{
    return tbm_mult_template(self, arg, kwarg, false);
}

static PyObject* tbm_mult_t(PyObject *self, PyObject *arg, PyObject *kwarg)
{
    return tbm_mult_template(self, arg, kwarg, true);
}

static PyObject* tbm_sprint(PyObject *self, PyObject *arg, PyObject *kwarg)
{
    PyArrayObject *arr_in;
    PyArrayObject *arr_str01; //!< two characters for 0 and 1
    uint32_t n_col;
    uint32_t n_row;
    char *format_str = NULL;

    static char *kwlist[] = {"in", "n_row", "n_col", "str01", "fmt", NULL};
    int ok = PyArg_ParseTupleAndKeywords(
        arg, kwarg, "O!IIO!|s", kwlist, &PyArray_Type, &arr_in, &n_row, &n_col, 
        &PyArray_Type, &arr_str01, &format_str);
    if (!ok)
        return PyErr_Format(PyExc_RuntimeError, "failed to parse parameters");
    if (arr_in == NULL) return NULL;

    if (PyArray_TYPE(arr_str01) != NPY_UINT8)
        return PyErr_Format(PyExc_RuntimeError, "4th arg must be uint8");
    if (PyArray_SIZE(arr_str01) < 2)
        return PyErr_Format(
            PyExc_RuntimeError, "4th arg shall have at least 2 characters");
    char *str01 = (char *)PyArray_DATA(arr_str01);

    bool use_gfnio;
    if ((format_str == NULL) || (strcmp(format_str, "default") == 0))
    {
        use_gfnio = false;
    }
    else if (strcmp(format_str, "gfni") == 0)
    {
        use_gfnio = true;
    }
    else
        return PyErr_Format(
            PyExc_RuntimeError, "format shall be 'gfni' or 'default'");

    // create output dimensions
    int n_dim = PyArray_NDIM(arr_in);
    npy_intp n_mat = PyArray_SIZE(arr_in);
    int n_dim_out;

    if (use_gfnio)
    {
        if (n_dim < 2)
            return PyErr_Format(
                PyExc_RuntimeError, "input need at least 2 dimensiond");
        uint32_t n_col8 = (n_col+7)/8; //!< number of columns in octets
        uint32_t n_row8 = (n_row+7)/8; //!< number of rows in octets
        npy_intp n_octet_col_in = PyArray_DIM(arr_in, n_dim-1);
        npy_intp n_octet_row_in = PyArray_DIM(arr_in, n_dim-2);
        if (n_col8 != n_octet_col_in)
            return PyErr_Format(
                PyExc_RuntimeError,
                "last dimension %d octets does not match n_col %d",
                n_octet_col_in, n_col);
        if (n_row8 != n_octet_row_in)
            return PyErr_Format(
                PyExc_RuntimeError,
                "last but one dimension %d octets does not match n_row %d",
                n_octet_row_in, n_row);
        n_dim_out = n_dim;
        n_mat /= n_octet_col_in*n_octet_row_in;
    }
    else
    {
        if (n_dim < 1)
            return failure(
                PyExc_RuntimeError, "input need at least 1 dimension");
        npy_intp n_bit_raw = PyArray_DIM(arr_in, n_dim-1);

        if ((n_col < 1) || (n_bit_raw < n_col))
            return failure(
                PyExc_RuntimeError,
                "n_col shall be inferior to the last dimension");
        if ((n_col < 1) || (32 < n_col))
            return failure(
                PyExc_RuntimeError,
                "last two dimensions shall be between 1 and 32 bits");
        n_dim_out = n_dim+1;
        n_mat /= n_bit_raw;

        npy_intp size_type = PyArray_ITEMSIZE(arr_in);
        if (n_bit_raw != 8*size_type)
            return failure(
                PyExc_RuntimeError,
                "last dimension shall be equal to the number of bits of the type");
    
    }

    npy_intp *out_dim = (npy_intp *)malloc(n_dim_out*sizeof(npy_intp));
    memcpy(out_dim, PyArray_DIMS(arr_in), n_dim*sizeof(npy_intp));
    out_dim[n_dim_out-2] = n_row;
    out_dim[n_dim_out-1] = n_col;
    PyObject *arr_out = PyArray_SimpleNew(n_dim_out, out_dim, NPY_UINT8);
    uint8_t *out = (uint8_t *)PyArray_DATA((PyArrayObject *)arr_out);

    // ensure the input array is contiguous.
    // PyArray_GETCONTIGUOUS will increase the reference count.
    arr_in = PyArray_GETCONTIGUOUS(arr_in);

    int py_type = PyArray_TYPE(arr_in);
    if (use_gfnio && (py_type == NPY_UINT64))
    {
        tbm_sprint8_gfnio(
            (uint64_t *)PyArray_DATA(arr_in), n_mat, n_row, n_col, str01, out);
    }
    else if ((py_type == NPY_INT8) || (py_type == NPY_UINT8))
    {
        bool special = (str01[0] == 0) && (str01[1] == -1);
        if (special)
        {
            printf("use special avx2 code!\n");
            tbm_sprint8_avx2(
                (uint8_t *)PyArray_DATA(arr_in), n_mat, str01, out);
        }
        else
            tbm_sprint8(
                (uint8_t *)PyArray_DATA(arr_in), n_mat, n_col, str01, out);
    }
    else if ((py_type == NPY_INT16) || (py_type == NPY_UINT16))
    {
        tbm_sprint16(
            (uint16_t *)PyArray_DATA(arr_in), n_mat, n_col, str01, out);
    }
    else if ((py_type == NPY_INT32) || (py_type == NPY_UINT32))
    {
        tbm_sprint32(
            (uint32_t *)PyArray_DATA(arr_in), n_mat, n_col, str01, out);
    }
    else
        failure(PyExc_RuntimeError, "input type is not supported");

        // decrease the reference count
    Py_DECREF(arr_in);
    free(out_dim);
    return arr_out;
}

static PyObject* tbm_transpose(PyObject *self, PyObject *arg, PyObject *kwarg)
{
    PyArrayObject *arr_in;
   char *method_str = NULL;

    static char *kwlist[] = {"in", "method", NULL};
    int ok = PyArg_ParseTupleAndKeywords(
        arg, kwarg, "O!|s", kwlist, &PyArray_Type, &arr_in, 
        &method_str);
    if (!ok)
        return PyErr_Format(PyExc_RuntimeError, "failed to parse parameters");
    if (arr_in == NULL) return NULL;

    bool use_gfnio;
    uint8_t i_fun = parse_method_str(method_str, &use_gfnio);
    if (i_fun == 0xff)
        return PyErr_Format(
            PyExc_RuntimeError,
            "method string shall be 'avx2', 'gfni', 'gfnio', or 'default'");

    // create output dimensions
    int n_dim = PyArray_NDIM(arr_in);
    npy_intp n_mat = PyArray_SIZE(arr_in);
    int n_dim_out = n_dim;
    npy_intp n_col8 = 1;
    npy_intp n_row8 = 1;

    if (use_gfnio)
    {
        if (n_dim < 2)
            return PyErr_Format(
                PyExc_RuntimeError, "input need at least 2 dimensiond");
        n_col8 = PyArray_DIM(arr_in, n_dim-1);
        n_row8 = PyArray_DIM(arr_in, n_dim-2);
        n_mat /= n_col8*n_row8;
    }
    else
    {
        if (n_dim < 1)
            return PyErr_Format(
                PyExc_RuntimeError, "input need at least 1 dimension");
        npy_intp n_bit_raw = PyArray_DIM(arr_in, n_dim-1);

        n_mat /= n_bit_raw;

        npy_intp size_type = PyArray_ITEMSIZE(arr_in);
        if (n_bit_raw != 8*size_type)
            return PyErr_Format(
                PyExc_RuntimeError,
                "last dimension shall be equal to the number of bits of the type");
    }

    npy_intp *out_dim = (npy_intp *)malloc(n_dim_out*sizeof(npy_intp));
    memcpy(out_dim, PyArray_DIMS(arr_in), n_dim*sizeof(npy_intp));
    if (use_gfnio)
    {
        out_dim[n_dim_out-1] = n_row8;
        out_dim[n_dim_out-2] = n_col8;
    }
    int py_type = PyArray_TYPE(arr_in);
    PyObject *arr_out = PyArray_SimpleNew(n_dim_out, out_dim, py_type);

    // ensure the input array is contiguous.
    // PyArray_GETCONTIGUOUS will increase the reference count.
    arr_in = PyArray_GETCONTIGUOUS(arr_in);
    free(out_dim);

    if (use_gfnio && (py_type == NPY_UINT64))
    {
        uint64_t *in = (uint64_t *)PyArray_DATA(arr_in);
        uint64_t *out = (uint64_t *)PyArray_DATA((PyArrayObject *)arr_out);
        tbm_transpose_gfnio(in, n_mat, n_row8, n_col8, out);
    }
    else if ((py_type == NPY_INT8) || (py_type == NPY_UINT8))
    {
        uint8_t *in = (uint8_t *)PyArray_DATA(arr_in);
        uint8_t *out = (uint8_t *)PyArray_DATA((PyArrayObject *)arr_out);
        tbm_1arg_int8_fun_t *fun[3] = {
            tbm_transpose8x8, tbm_transpose8x8_avx2, tbm_transpose8x8_gfni};
        fun[i_fun](in, n_mat, out);
    }
    else if ((py_type == NPY_INT16) || (py_type == NPY_UINT16))
    {
        uint16_t *in = (uint16_t *)PyArray_DATA(arr_in);
        uint16_t *out = (uint16_t *)PyArray_DATA((PyArrayObject *)arr_out);
        tbm_1arg_int16_fun_t *fun[3] = {
            tbm_transpose16x16, tbm_transpose16x16_avx2, 
            tbm_transpose16x16_gfni};
        fun[i_fun](in, n_mat, out);
    }
    else if ((py_type == NPY_INT32) || (py_type == NPY_UINT32))
    {
        uint32_t *in = (uint32_t *)PyArray_DATA(arr_in);
        uint32_t *out = (uint32_t *)PyArray_DATA((PyArrayObject *)arr_out);
        tbm_1arg_int32_fun_t *fun[3] = {
            tbm_transpose32x32, tbm_transpose32x32_avx2, 
            tbm_transpose32x32_gfni};
        fun[i_fun](in, n_mat, out);
    }
    else
        PyErr_Format(PyExc_RuntimeError, "input type is not supported");

    // decrease the reference count
    Py_DECREF(arr_in);
    return arr_out;
}

//______________________________________________________________________________
// set up the methods table
static PyMethodDef method_def[] = {
    {
        "encode", (PyCFunction)tbm_encode, METH_VARARGS | METH_KEYWORDS,
        "encode a square matrix into a tinybinmat"
    },
    {
        "mult", (PyCFunction)tbm_mult, METH_VARARGS | METH_KEYWORDS, 
        "multiply twp tinybinmat matrices"
    },
    {
        "mult_t", (PyCFunction)tbm_mult_t, METH_VARARGS | METH_KEYWORDS,
        "multiply a tinybinmat by another transposed tinybinmat"
    },
    {
        "sprint", (PyCFunction)tbm_sprint, METH_VARARGS | METH_KEYWORDS, 
        "convert to uint8 array"
    },
    {
        "transpose", (PyCFunction)tbm_transpose, METH_VARARGS | METH_KEYWORDS, 
        "transpose tinybinmat"
    },
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
    PyObject *m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    import_array();
    return m;
}
