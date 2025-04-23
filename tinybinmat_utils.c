#include "tinybinmat_utils.h"

void *failure(PyObject *type, const char *message)
{
    PyErr_SetString(type, message);
    return NULL;
}
