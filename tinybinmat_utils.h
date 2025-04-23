#if !defined(_TINYBITMAT_UTILS)
#define _TINYBITMAT_UTILS
#include "Python.h"
#include "math.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "numpy/ndarraytypes.h"
#include "immintrin.h"
#include "stdbool.h"

#ifdef _MSC_VER
#define __UNUSED__
#else
#define __UNUSED__ __attribute__((unused))
#endif

extern void *failure(PyObject *type, const char *message);
#endif
