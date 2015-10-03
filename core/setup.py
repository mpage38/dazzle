# python setup.py build_ext --inplace

# distutils uses the Makefile distutils.sysconfig.get_makefile_filename()
# for compiling and linking: a sea of options.

# http://docs.python.org/distutils/introduction.html
# http://docs.python.org/distutils/apiref.html  20 pages ...
# http://stackoverflow.com/questions/tagged/distutils+python

import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
# from Cython.Build import cythonize


ext_modules = [Extension(
    name="chunk_evaluator",
    sources=["cython_chunk_evaluator.pyx", "chunk_evaluator.c"],
        # extra_objects=["fc.o"],  # if you compile fc.c separately
    include_dirs = [numpy.get_include()],  # .../site-packages/numpy/core/include
    language="c",
        # libraries=
        # extra_compile_args = "...".split(),
        # extra_link_args = "...".split()
    )]

setup(
    name = 'chunk_evaluator',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
        # ext_modules = cythonize(ext_modules)  ? not in 0.14.1
    # version=
    # description=
    # author=
    # author_email=
    )