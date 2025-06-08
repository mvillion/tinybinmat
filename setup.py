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

cflag = ["-O3", "-mavx2"]
lib_list = []
for lib_name in ["tinybinmat", "tinybinmat_avx2"]:
    lib_list.append((lib_name, src_from_name(lib_name, cflag=cflag)))

cflag = ["-O3", "-mavx2", "-mgfni", "-DUSE_GFNI"]
for lib_name in ["tinybinmat_gfnio"]:
    lib_list.append((lib_name, src_from_name(lib_name, cflag=cflag)))
# cflag = ["-O3", "-mavx2", "-mgfni", "-DUSE_GFNI"]
# lib_list.append(
#     ("tinybinmat_gfnio", src_from_name("tinybinmat_avx2", cflag=cflag)))

lib_list += [
    ("tinybinmat_utils", {"sources": ["tinybinmat_utils.c"]} | inc_dir),
]

setup(
    libraries=lib_list,
    ext_modules=[
        Extension(
            name="tinybinmat", sources=["tinybinmat_py.c"],
            extra_compile_args=["-mavx2"],
            include_dirs=[this_file_path, numpy.get_include()],
            libraries=[k[0] for k in lib_list],
        ),
    ],
    setup_requires=["setuptools-git-versioning"]
)
