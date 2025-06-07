import numpy
import os

from setuptools import Extension, setup
from sysconfig import get_paths

this_file_path = os.path.dirname(__file__)
if this_file_path == "":
    this_file_path = "."
print("coucou %s" % this_file_path)
inc_dir = {
    "include_dirs": [
        this_file_path, numpy.get_include(), get_paths()["include"],
    ]}


def src_from_name(name, cflag=None):
    out = {
        "sources": ["%s.c" % name],
    }
    if cflag is not None:
        if type(cflag) is not list:
            cflag = [cflag]
        out["cflags"] = cflag
    return out

cflag = ["-O3", "-mavx2", "-mgfni"]
lib_list = [
    ("tinybinmat", src_from_name("tinybinmat", cflag=cflag)),
    # ("tinybinmat_avx2", src_from_name("tinybinmat_avx2", cflag=cflag)),
    # ("tinybinmat_gfni", src_from_name("tinybinmat_gfni", cflag=cflag)),
    ("tinybinmat_gfnio", src_from_name("tinybinmat_gfnio", cflag=cflag)),
    ("tinybinmat_utils", {"sources": ["tinybinmat_utils.c"]} | inc_dir),
]

setup(
    libraries=lib_list,
    ext_modules=[
        Extension(
            name="tinybinmat", sources=["tinybinmat_py.c"],
            extra_compile_args=["-mavx2", "-mgfni"],
            include_dirs=[this_file_path, numpy.get_include()],
            libraries=[k[0] for k in lib_list],
        ),
    ],
    setup_requires=["setuptools-git-versioning"]
)
