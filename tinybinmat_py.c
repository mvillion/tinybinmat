#include "tinybinmat_utils.h"

#include "tinybinmat.c"

//______________________________________________________________________________
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
    if (py_type == NPY_UINT8)
    {
        tbm_sprint8((uint8_t *)PyArray_DATA(arr_in), n_mat, n_bit, str01, out);
    }
    else if (py_type == NPY_UINT16)
    {
        tbm_sprint16(
            (uint16_t *)PyArray_DATA(arr_in), n_mat, n_bit, str01, out);
    }

    // decrease the reference count
    Py_DECREF(arr_in);
    free(out_dim);
    return arr_out;
}

static PyObject* tbm_transpose(PyObject *self, PyObject *arg)
{
    PyArrayObject *arr_in;
    uint32_t n_bit;

    int ok = PyArg_ParseTuple(arg, "O!I", &PyArray_Type, &arr_in, &n_bit);
    if (!ok)
        return failure(PyExc_RuntimeError, "failed to parse parameters");
    if (arr_in == NULL) return NULL;

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

    PyObject *arr_out = PyArray_NewLikeArray(arr_in, NPY_ANYORDER, NULL, 0);
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
    if (py_type == NPY_UINT8)
    {
        memcpy(out, PyArray_DATA(arr_in), n_mat*n_bit_raw*sizeof(uint8_t));
    }
    else if (py_type == NPY_UINT16)
    {
        memcpy(out, PyArray_DATA(arr_in), n_mat*n_bit_raw*sizeof(uint16_t));
    }

    // decrease the reference count
    Py_DECREF(arr_in);
    return arr_out;
}

//______________________________________________________________________________
// set up the methods table
static PyMethodDef method_def[] = {
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
