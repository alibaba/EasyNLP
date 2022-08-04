# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals
from distutils.core import setup
from Cython.Build import cythonize
import numpy
from distutils.extension import Extension
from subprocess import check_output
from distutils import sysconfig
import sys

extra_compile_args = ['-std=c++11']
extra_link_args = []

# Super hacky way of determining if clang or gcc is being used
CC = sysconfig.get_config_vars().get('CC', 'gcc').split(' ')[0]
out = check_output([CC, '--version'])
if sys.platform == 'darwin':
    extra_compile_args = ["-stdlib=libc++"]
    extra_link_args=['-stdlib=libc++']

extensions = [
    Extension(
        "hype.graph_dataset",
        ["hype/graph_dataset.pyx"],
        include_dirs=[numpy.get_include()],
        extra_link_args=extra_link_args,
        extra_compile_args=extra_compile_args,
        language='c++',
    ),
    Extension(
        "hype.adjacency_matrix_dataset",
        ["hype/adjacency_matrix_dataset.pyx"],
        include_dirs=[numpy.get_include()],
        extra_link_args=extra_link_args,
        extra_compile_args=extra_compile_args,
        language='c++',
    ),
]


setup(
    ext_modules=cythonize(extensions),
)
