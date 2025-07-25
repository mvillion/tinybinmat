#include "tinybinmat.h"
#include "tinybinmat_conv.h"
#include "tinybinmat_perm.h"
#include "tinybinmat_avx2.h"
#include "tinybinmat_utils.h"

#if defined(__aarch64__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif

#define N_METHOD 4
uint8_t parse_method_str(const char *method_str)
{
#if defined(__aarch64__)
    long hwcaps = getauxval(AT_HWCAP);
#endif
    uint8_t i_fun = 0xff;
    if ((method_str == NULL) || (strcmp(method_str, "default") == 0))
    {
        i_fun = 0;
    }
    else if (strcmp(method_str, "avx2") == 0)
    {
#if defined(__x86_64__) || defined(_M_X64)
        i_fun = __builtin_cpu_supports("avx2") ? 1 : 0xff;
#endif
    }
    else if (strcmp(method_str, "gfni") == 0)
    {
#if defined(__x86_64__) || defined(_M_X64)
        i_fun = __builtin_cpu_supports("gfni") ? 2 : 0xff;
#endif
    }
    else if (strcmp(method_str, "simd") == 0)
    {
#if defined(__aarch64__)
        i_fun = (hwcaps & HWCAP_ASIMD) ? 3 : 0xff;
#endif
    }
    else
        PyErr_Format(
            PyExc_RuntimeError,
            "method string shall be 'avx2', 'gfni', 'simd' or 'default'");
    if (i_fun == 0xff)
        PyErr_Format(
            PyExc_RuntimeError,
            "%s instruction set is not supported by the CPU", method_str);
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
        return PyErr_Format(PyExc_RuntimeError, "failed to parse parameters");
    if (arr_in == NULL) return NULL;

    if ((format_str == NULL) || (strcmp(format_str, "default") == 0))
    {
    }
    else
        return PyErr_Format(
            PyExc_RuntimeError, "format shall be 'default' (gfni)");

    int n_dim = PyArray_NDIM(arr_in);
    if (n_dim < 2)
        return PyErr_Format(
            PyExc_RuntimeError, "input need at least 2 dimensions");
    npy_intp n_col = PyArray_DIM(arr_in, n_dim-1);
    npy_intp n_row = PyArray_DIM(arr_in, n_dim-2);

    npy_intp size_type = PyArray_ITEMSIZE(arr_in);
    if (size_type != 1)
        return PyErr_Format(
            PyExc_RuntimeError, "input type size shall be equal to 1 octet");

    uint64_t n_mat = (uint64_t)(PyArray_SIZE(arr_in)/n_col/n_row);

    int py_type;
    int n_dim_out;

    py_type = NPY_UINT64;
    n_dim_out = n_dim;

    // create output dimensions
    npy_intp *out_dim = (npy_intp *)malloc(n_dim_out*sizeof(npy_intp));
    memcpy(out_dim, PyArray_DIMS(arr_in), (n_dim_out-1)*sizeof(npy_intp));
    out_dim[n_dim_out-1] = (n_col+7)/8;
    out_dim[n_dim_out-2] = (n_row+7)/8;
    PyObject *arr_out = PyArray_SimpleNew(n_dim_out, out_dim, py_type);

    // ensure the input array is contiguous.
    // PyArray_GETCONTIGUOUS will increase the reference count.
    arr_in = PyArray_GETCONTIGUOUS(arr_in);
    free(out_dim);

    uint8_t *in = (uint8_t *)PyArray_DATA(arr_in);
    tbm_encode_gfnio(
        in, n_mat, n_row, n_col,
        (uint64_t *)PyArray_DATA((PyArrayObject *)arr_out));

    // decrease the reference count
    Py_DECREF(arr_in);
    return arr_out;
}

static PyObject* tbm_eye(PyObject *self, PyObject *arg)
{
    uint64_t n_mat;
    uint32_t n_row_col;
    int ok = PyArg_ParseTuple(arg, "IK", &n_row_col, &n_mat);
    if (!ok)
        return PyErr_Format(PyExc_RuntimeError, "failed to parse parameters");

    int py_type = NPY_UINT64;
    int n_dim_out = 3;

    // create output dimensions
    uint32_t n_dim_o = (n_row_col+7)/8; //!< number of rows/columns in octets
    npy_intp *out_dim = (npy_intp *)malloc(n_dim_out*sizeof(npy_intp));
    out_dim[0] = n_mat;
    out_dim[1] = n_dim_o;
    out_dim[2] = n_dim_o;
    PyObject *arr_out = PyArray_SimpleNew(n_dim_out, out_dim, py_type);
    free(out_dim);

    uint64_t *out = (uint64_t *)PyArray_DATA((PyArrayObject *)arr_out);
    if (0 < n_mat)
        tbm_eye_u64(n_row_col, out);
    for (uint64_t i_mat = 1; i_mat < n_mat; i_mat++)
        memcpy(
            out+i_mat*n_dim_o*n_dim_o, out, n_dim_o*n_dim_o*sizeof(uint64_t));
    return arr_out;
}

static PyObject* tbm_circul(PyObject *self, PyObject *arg)
{
    PyArrayObject *arr_in; //!< 1st array of matrices to multiply
    uint32_t n_row_col;

    int ok = PyArg_ParseTuple(arg, "O!I", &PyArray_Type, &arr_in, &n_row_col);
    if (!ok)
        return PyErr_Format(PyExc_RuntimeError, "failed to parse parameters");
    if (arr_in == NULL) return NULL;

    int n_dim = PyArray_NDIM(arr_in);
    if (n_dim < 1)
        return PyErr_Format(
            PyExc_RuntimeError, "input need at least 1 dimension");
    npy_intp n_dim_o_in = PyArray_DIM(arr_in, n_dim-1);
    uint32_t n_dim_o = (n_row_col+7)/8; //!< number of rows/columns in octets
    if (n_dim_o != n_dim_o_in)
        return PyErr_Format(
            PyExc_RuntimeError,
            "last dimension %d octets does not match n_row/n_col %d",
            n_dim_o_in, n_row_col);

    npy_intp size_type = PyArray_ITEMSIZE(arr_in);
    if (size_type != 1)
        return PyErr_Format(
            PyExc_RuntimeError, "input type size shall be equal to 1 octet");

    uint64_t n_mat = (uint64_t)(PyArray_SIZE(arr_in)/n_dim_o);

    int py_type = NPY_UINT64;
    int n_dim_out = n_dim+1;

    // create output dimensions
    npy_intp *out_dim = (npy_intp *)malloc(n_dim_out*sizeof(npy_intp));
    memcpy(out_dim, PyArray_DIMS(arr_in), (n_dim_out-1)*sizeof(npy_intp));
    out_dim[n_dim_out-1] = n_dim_o;
    out_dim[n_dim_out-2] = n_dim_o;
    PyObject *arr_out = PyArray_SimpleNew(n_dim_out, out_dim, py_type);

    // ensure the input array is contiguous.
    // PyArray_GETCONTIGUOUS will increase the reference count.
    arr_in = PyArray_GETCONTIGUOUS(arr_in);
    free(out_dim);

    uint8_t *in = (uint8_t *)PyArray_DATA(arr_in);
    printf("tbm_circul n_row_col %u n_mat %lu\n", n_row_col, n_mat);
    tbm_circul_u64(
        in, n_mat, n_row_col,
        (uint64_t *)PyArray_DATA((PyArrayObject *)arr_out));

    // decrease the reference count
    Py_DECREF(arr_in);
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

    uint8_t i_fun = parse_method_str(method_str);
    if (i_fun == 0xff)
        return NULL;
    i_fun += is_transposed ? N_METHOD : 0;

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

    if (n_mat != n_mat2)
        return PyErr_Format(
            PyExc_RuntimeError, "two arrays shall have the same size");

    npy_intp *out_dim = (npy_intp *)malloc(n_dim_out*sizeof(npy_intp));
    memcpy(out_dim, PyArray_DIMS(arr_in), n_dim*sizeof(npy_intp));
    out_dim[n_dim_out-1] = is_transposed ? n_row8_2 : n_col8_2;
    out_dim[n_dim_out-2] = n_row8;
    int py_type = PyArray_TYPE(arr_in);
    PyObject *arr_out = PyArray_SimpleNew(n_dim_out, out_dim, py_type);

    // ensure the input array is contiguous.
    // PyArray_GETCONTIGUOUS will increase the reference count.
    arr_in = PyArray_GETCONTIGUOUS(arr_in);
    arr_in2 = PyArray_GETCONTIGUOUS(arr_in2);
    free(out_dim);

    if (py_type == NPY_UINT64)
    {
        uint64_t *in = (uint64_t *)PyArray_DATA(arr_in);
        uint64_t *in2 = (uint64_t *)PyArray_DATA(arr_in2);
        uint64_t *out = (uint64_t *)PyArray_DATA((PyArrayObject *)arr_out);
        tbm_mult_fun_t *fun[N_METHOD*2] = {
            tbm_mult_u64, tbm_mult_avx2, tbm_mult_gfni, tbm_mult_simd,
            tbm_mult_t_u64, tbm_mult_t_avx2, tbm_mult_t_gfni, tbm_mult_t_simd};
        if (is_transposed)
            fun[i_fun](in, n_mat, n_row8, n_col8, in2, n_row8_2, out);
        else
            fun[i_fun](in, n_mat, n_row8, n_col8, in2, n_col8_2, out);
    }
    else
        PyErr_Format(PyExc_RuntimeError, "input type is not supported");

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

    if ((format_str == NULL) || (strcmp(format_str, "default") == 0))
    {
    }
    else
        return PyErr_Format(
            PyExc_RuntimeError, "format shall be 'default' (gfni)");

    // create output dimensions
    int n_dim = PyArray_NDIM(arr_in);
    npy_intp n_mat = PyArray_SIZE(arr_in);
    int n_dim_out;

    if (n_dim < 2)
        return PyErr_Format(
            PyExc_RuntimeError, "input need at least 2 dimensions");
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

    npy_intp *out_dim = (npy_intp *)malloc(n_dim_out*sizeof(npy_intp));
    memcpy(out_dim, PyArray_DIMS(arr_in), n_dim*sizeof(npy_intp));
    out_dim[n_dim_out-2] = n_row;
    out_dim[n_dim_out-1] = n_col;
    PyObject *arr_out = PyArray_SimpleNew(n_dim_out, out_dim, NPY_UINT8);
    uint8_t *out = (uint8_t *)PyArray_DATA((PyArrayObject *)arr_out);

    // ensure the input array is contiguous.
    // PyArray_GETCONTIGUOUS will increase the reference count.
    arr_in = PyArray_GETCONTIGUOUS(arr_in);
    free(out_dim);

    int py_type = PyArray_TYPE(arr_in);
    if (py_type == NPY_UINT64)
    {
        tbm_sprint8_gfnio(
            (uint64_t *)PyArray_DATA(arr_in), n_mat, n_row, n_col, str01, out);
    }
    else
        PyErr_Format(PyExc_RuntimeError, "input type is not supported");

        // decrease the reference count
    Py_DECREF(arr_in);
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

    uint8_t i_fun = parse_method_str(method_str);
    if (i_fun == 0xff)
        return NULL;

    // create output dimensions
    int n_dim = PyArray_NDIM(arr_in);
    npy_intp n_mat = PyArray_SIZE(arr_in);
    int n_dim_out = n_dim;
    npy_intp n_col8 = 1;
    npy_intp n_row8 = 1;

    if (n_dim < 2)
        return PyErr_Format(
            PyExc_RuntimeError, "input need at least 2 dimensions");
    n_col8 = PyArray_DIM(arr_in, n_dim-1);
    n_row8 = PyArray_DIM(arr_in, n_dim-2);
    n_mat /= n_col8*n_row8;

    npy_intp *out_dim = (npy_intp *)malloc(n_dim_out*sizeof(npy_intp));
    memcpy(out_dim, PyArray_DIMS(arr_in), n_dim*sizeof(npy_intp));
    out_dim[n_dim_out-1] = n_row8;
    out_dim[n_dim_out-2] = n_col8;
    int py_type = PyArray_TYPE(arr_in);
    PyObject *arr_out = PyArray_SimpleNew(n_dim_out, out_dim, py_type);

    // ensure the input array is contiguous.
    // PyArray_GETCONTIGUOUS will increase the reference count.
    arr_in = PyArray_GETCONTIGUOUS(arr_in);
    free(out_dim);

    if (py_type == NPY_UINT64)
    {
        uint64_t *in = (uint64_t *)PyArray_DATA(arr_in);
        uint64_t *out = (uint64_t *)PyArray_DATA((PyArrayObject *)arr_out);
        tbm_transpose_fun_t *fun[N_METHOD] = {
            tbm_transpose_u64, tbm_transpose_avx2, tbm_transpose_gfni,
            tbm_transpose_simd
        };
        fun[i_fun](in, n_mat, n_row8, n_col8, out);
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
        "circul", tbm_circul, METH_VARARGS,
        "create circulant matrix in tinybinmat gfni format"
    },
    {
        "encode", (PyCFunction)tbm_encode, METH_VARARGS | METH_KEYWORDS,
        "encode a square matrix into a tinybinmat"
    },
    {
        "eye", tbm_eye, METH_VARARGS,
        "create eye matrix in tinybinmat gfni format"
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
