#include "tinybinmat.h"
#include "tinybinmat_utils.h"

//______________________________________________________________________________
static PyObject* tbm_encode(PyObject *self, PyObject *arg)
{
    PyArrayObject *arr_in; //!< 1st array of matrices to multiply

    int ok = PyArg_ParseTuple(arg, "O!", &PyArray_Type, &arr_in);
    if (!ok)
        return failure(PyExc_RuntimeError, "failed to parse parameters");
    if (arr_in == NULL) return NULL;

    int n_dim = PyArray_NDIM(arr_in);
    if (n_dim < 2)
        return failure(PyExc_RuntimeError, "input need at least 2 dimension");
    npy_intp n_bit = PyArray_DIM(arr_in, n_dim-1);

    if (n_bit != PyArray_DIM(arr_in, n_dim-2))
        return failure(
            PyExc_RuntimeError, "last two dimensions shall be equal");

    npy_intp size_type = PyArray_ITEMSIZE(arr_in);
    if (size_type != 1)
        return failure(
            PyExc_RuntimeError, "input type size shall be equal to 1 octet");

    uint64_t n_mat = (uint64_t)(PyArray_SIZE(arr_in)/n_bit/n_bit);

    int py_type;
    uint8_t n_bit_raw;
    if (n_bit <= 8)
    {
        py_type = NPY_UINT8; n_bit_raw = 8;
    }
    else if (n_bit <= 16)
    {
        py_type = NPY_UINT16; n_bit_raw = 16;
    }
    else if (n_bit <= 32)
    {
        py_type = NPY_UINT32; n_bit_raw = 32;
    }
    else
        return failure(
            PyExc_RuntimeError,
            "input type size shall be equal to 1, 2, or 4 octets");

    // create output dimensions
    npy_intp *out_dim = (npy_intp *)malloc((n_dim-1)*sizeof(npy_intp));
    memcpy(out_dim, PyArray_DIMS(arr_in), (n_dim-2)*sizeof(npy_intp));
    out_dim[n_dim-2] = n_bit_raw;
    PyObject *arr_out = PyArray_SimpleNew(n_dim-1, out_dim, py_type);

    // ensure the input array is contiguous.
    // PyArray_GETCONTIGUOUS will increase the reference count.
    arr_in = PyArray_GETCONTIGUOUS(arr_in);

    uint8_t *in = (uint8_t *)PyArray_DATA(arr_in);
    if (py_type == NPY_UINT8)
    {
        tbm_encode8(
            in, n_mat, n_bit, n_bit_raw,
            (uint8_t *)PyArray_DATA((PyArrayObject *)arr_out));
    }
    else if (py_type == NPY_UINT16)
    {
        tbm_encode16(
            in, n_mat, n_bit, n_bit_raw,
            (uint16_t *)PyArray_DATA((PyArrayObject *)arr_out));
    }
    else if (py_type == NPY_UINT32)
    {
        tbm_encode32(
            in, n_mat, n_bit, n_bit_raw,
            (uint32_t *)PyArray_DATA((PyArrayObject *)arr_out));
    }

    // decrease the reference count
    Py_DECREF(arr_in);
    return arr_out;
}

static PyObject* tbm_mult_template(
    PyObject *self, PyObject *arg, bool is_transposed)
{
    PyArrayObject *arr_in; //!< 1st array of matrices to multiply
    PyArrayObject *arr_in2; //!< 2nd array of matrices to multiply (transposed)

    int ok = PyArg_ParseTuple(
        arg, "O!O!", &PyArray_Type, &arr_in, &PyArray_Type, &arr_in2);
    if (!ok)
        return failure(PyExc_RuntimeError, "failed to parse parameters");
    if (arr_in == NULL) return NULL;

    int n_dim = PyArray_NDIM(arr_in);
    if (n_dim < 1)
        return failure(PyExc_RuntimeError, "input need at least 1 dimension");
    npy_intp n_bit_raw = PyArray_DIM(arr_in, n_dim-1);

    npy_intp size_type = PyArray_ITEMSIZE(arr_in);
    if (n_bit_raw != 8*size_type)
        return failure(
            PyExc_RuntimeError,
            "last dimension shall be equal to the number of bits of the type");

    if (PyArray_NBYTES(arr_in2) != PyArray_NBYTES(arr_in))
        return failure(
            PyExc_RuntimeError, "two arrays shall have the same size");

    uint64_t n_mat = (uint64_t)(PyArray_SIZE(arr_in)/n_bit_raw);

    // create output dimensions
    PyObject *arr_out = PyArray_NewLikeArray(arr_in, NPY_ANYORDER, NULL, 0);

    // ensure the input array is contiguous.
    // PyArray_GETCONTIGUOUS will increase the reference count.
    arr_in = PyArray_GETCONTIGUOUS(arr_in);
    arr_in2 = PyArray_GETCONTIGUOUS(arr_in2);

    int py_type = PyArray_TYPE(arr_in);
    if ((py_type == NPY_INT8) || (py_type == NPY_UINT8))
    {
        uint8_t *in = (uint8_t *)PyArray_DATA(arr_in);
        uint8_t *in2 = (uint8_t *)PyArray_DATA(arr_in2);
        uint8_t *out = (uint8_t *)PyArray_DATA((PyArrayObject *)arr_out);
        if (is_transposed)
            tbm_mult_t8x8(in, in2, n_mat, out);
        else
            tbm_mult8x8(in, in2, n_mat, out);
    }
    else if ((py_type == NPY_INT16) || (py_type == NPY_UINT16))
    {
        uint16_t *in = (uint16_t *)PyArray_DATA(arr_in);
        uint16_t *in2 = (uint16_t *)PyArray_DATA(arr_in2);
        uint16_t *out = (uint16_t *)PyArray_DATA((PyArrayObject *)arr_out);
        if (is_transposed)
            tbm_mult_t16x16(in, in2, n_mat, out);
        else
            tbm_mult16x16(in, in2, n_mat, out);
    }
    else if ((py_type == NPY_INT32) || (py_type == NPY_UINT32))
    {
        uint32_t *in = (uint32_t *)PyArray_DATA(arr_in);
        uint32_t *in2 = (uint32_t *)PyArray_DATA(arr_in2);
        uint32_t *out = (uint32_t *)PyArray_DATA((PyArrayObject *)arr_out);
        if (is_transposed)
            tbm_mult_t32x32(in, in2, n_mat, out);
        else
            tbm_mult32x32(in, in2, n_mat, out);
    }
    else
        failure(PyExc_RuntimeError, "input type is not supported");

    // decrease the reference count
    Py_DECREF(arr_in2);
    Py_DECREF(arr_in);
    return arr_out;
}

static PyObject* tbm_mult(PyObject *self, PyObject *arg)
{
    return tbm_mult_template(self, arg, false);
}

static PyObject* tbm_mult_t(PyObject *self, PyObject *arg)
{
    return tbm_mult_template(self, arg, true);
}

static PyObject* tbm_print(PyObject *self, PyObject *arg)
{
    PyArrayObject *arr_in;
    char *str01; //!< two characters for 0 and 1
    uint32_t n_bit;

    int ok = PyArg_ParseTuple(
        arg, "O!Is", &PyArray_Type, &arr_in, &n_bit, &str01);
    if (!ok)
        return failure(PyExc_RuntimeError, "failed to parse parameters");
    if (arr_in == NULL) return NULL;

    if (strlen(str01) < 2)
        return failure(PyExc_RuntimeError, "need at least 2 characters");

    int n_dim = PyArray_NDIM(arr_in);
    if (n_dim < 1)
        return failure(PyExc_RuntimeError, "input need at least 1 dimension");
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

    npy_intp size_type = PyArray_ITEMSIZE(arr_in);
    if (n_bit_raw != 8*size_type)
        return failure(
            PyExc_RuntimeError,
            "last dimension shall be equal to the number of bits of the type");

    int py_type = PyArray_TYPE(arr_in);
    if (py_type == NPY_UINT8)
    {
        uint8_t *mat_list = (uint8_t *)PyArray_DATA(arr_in);
        tbm_print8(mat_list, n_mat, n_bit, str01);
    }
    else if (py_type == NPY_UINT16)
    {
        uint16_t *mat_list = (uint16_t *)PyArray_DATA(arr_in);
        tbm_print16(mat_list, n_mat, n_bit, str01);
    }

    // decrease the reference count
    Py_DECREF(arr_in);
    Py_RETURN_NONE;
}

static PyObject* tbm_sprint(PyObject *self, PyObject *arg)
{
    PyArrayObject *arr_in;
    PyArrayObject *arr_str01; //!< two characters for 0 and 1
    uint32_t n_bit;

    int ok = PyArg_ParseTuple(
        arg, "O!IO!", &PyArray_Type, &arr_in, &n_bit, &PyArray_Type,
        &arr_str01);
    if (!ok)
        return failure(PyExc_RuntimeError, "failed to parse parameters");
    if (arr_in == NULL) return NULL;

    if (PyArray_TYPE(arr_str01) != NPY_UINT8)
        return failure(PyExc_RuntimeError, "3rd arg must be uint8");
    if (PyArray_SIZE(arr_str01) < 2)
        return failure(
            PyExc_RuntimeError, "3rd arg shall have at least 2 characters");
    char *str01 = (char *)PyArray_DATA(arr_str01);
    bool special = (str01[0] == 0) && (str01[1] == -1);

    // create output dimensions
    int n_dim = PyArray_NDIM(arr_in);
    if (n_dim < 1)
        return failure(PyExc_RuntimeError, "input need at least 1 dimension");
    npy_intp n_bit_raw = PyArray_DIM(arr_in, n_dim-1);

    if ((n_bit < 1) || (n_bit_raw < n_bit))
        return failure(
            PyExc_RuntimeError,
            "n_bit shall be inferior to the last dimension");
    npy_intp n_mat = PyArray_SIZE(arr_in)/n_bit_raw;

    npy_intp *out_dim = (npy_intp *)malloc((n_dim+1)*sizeof(npy_intp));
    memcpy(out_dim, PyArray_DIMS(arr_in), (n_dim-1)*sizeof(npy_intp));
    out_dim[n_dim-1] = n_bit;
    out_dim[n_dim] = n_bit;
    PyObject *arr_out = PyArray_SimpleNew(n_dim+1, out_dim, NPY_UINT8);
    uint8_t *out = (uint8_t *)PyArray_DATA((PyArrayObject *)arr_out);

    // ensure the input array is contiguous.
    // PyArray_GETCONTIGUOUS will increase the reference count.
    arr_in = PyArray_GETCONTIGUOUS(arr_in);

    npy_intp size_type = PyArray_ITEMSIZE(arr_in);
    if (n_bit_raw != 8*size_type)
        return failure(
            PyExc_RuntimeError,
            "last dimension shall be equal to the number of bits of the type");

    int py_type = PyArray_TYPE(arr_in);
    if ((py_type == NPY_INT8) || (py_type == NPY_UINT8))
    {
        if (special)
        {
            printf("use special avx2 code!\n");
            tbm_sprint8_avx2(
                (uint8_t *)PyArray_DATA(arr_in), n_mat, str01, out);
        }
        else
            tbm_sprint8(
                (uint8_t *)PyArray_DATA(arr_in), n_mat, n_bit, str01, out);
    }
    else if ((py_type == NPY_INT16) || (py_type == NPY_UINT16))
    {
        tbm_sprint16(
            (uint16_t *)PyArray_DATA(arr_in), n_mat, n_bit, str01, out);
    }
    else if ((py_type == NPY_INT32) || (py_type == NPY_UINT32))
    {
        tbm_sprint32(
            (uint32_t *)PyArray_DATA(arr_in), n_mat, n_bit, str01, out);
    }
    else
        failure(PyExc_RuntimeError, "input type is not supported");

        // decrease the reference count
    Py_DECREF(arr_in);
    free(out_dim);
    return arr_out;
}

static PyObject* tbm_transpose(PyObject *self, PyObject *arg)
{
    PyArrayObject *arr_in;

    int ok = PyArg_ParseTuple(arg, "O!", &PyArray_Type, &arr_in);
    if (!ok)
        return failure(PyExc_RuntimeError, "failed to parse parameters");
    if (arr_in == NULL) return NULL;

    // create output dimensions
    int n_dim = PyArray_NDIM(arr_in);
    if (n_dim < 1)
        return failure(PyExc_RuntimeError, "input need at least 1 dimension");
    npy_intp n_bit_raw = PyArray_DIM(arr_in, n_dim-1);

    uint64_t n_mat = (uint64_t)(PyArray_SIZE(arr_in)/n_bit_raw);

    PyObject *arr_out = PyArray_NewLikeArray(arr_in, NPY_ANYORDER, NULL, 0);

    // ensure the input array is contiguous.
    // PyArray_GETCONTIGUOUS will increase the reference count.
    arr_in = PyArray_GETCONTIGUOUS(arr_in);

    npy_intp size_type = PyArray_ITEMSIZE(arr_in);
    if (n_bit_raw != 8*size_type)
        return failure(
            PyExc_RuntimeError,
            "last dimension shall be equal to the number of bits of the type");

    int py_type = PyArray_TYPE(arr_in);
    if ((py_type == NPY_INT8) || (py_type == NPY_UINT8))
    {
        uint8_t *in = (uint8_t *)PyArray_DATA(arr_in);
        uint8_t *out = (uint8_t *)PyArray_DATA((PyArrayObject *)arr_out);
        tbm_transpose8x8(in, n_mat, out);
    }
    else if ((py_type == NPY_INT16) || (py_type == NPY_UINT16))
    {
        uint16_t *in = (uint16_t *)PyArray_DATA(arr_in);
        uint16_t *out = (uint16_t *)PyArray_DATA((PyArrayObject *)arr_out);
        tbm_transpose16x16(in, n_mat, out);
    }
    else if ((py_type == NPY_INT32) || (py_type == NPY_UINT32))
    {
        uint32_t *in = (uint32_t *)PyArray_DATA(arr_in);
        uint32_t *out = (uint32_t *)PyArray_DATA((PyArrayObject *)arr_out);
        tbm_transpose32x32(in, n_mat, out);
    }
    else
        failure(PyExc_RuntimeError, "input type is not supported");

    // decrease the reference count
    Py_DECREF(arr_in);
    return arr_out;
}

//______________________________________________________________________________
// set up the methods table
static PyMethodDef method_def[] = {
    {"encode", tbm_encode, METH_VARARGS,
        "encode a square matrix into a tinybinmat"},
    {"mult", tbm_mult, METH_VARARGS, "multiply twp tinybinmat matrices"},
    {"mult_t", tbm_mult_t, METH_VARARGS,
    "multiply a tinybinmat by another transposed tinybinmat"},
    {"print", tbm_print, METH_VARARGS, "print tinybinmat"},
    {"sprint", tbm_sprint, METH_VARARGS, "convert to uint8 array"},
    {"transpose", tbm_transpose, METH_VARARGS, "transpose tinybinmat"},
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
