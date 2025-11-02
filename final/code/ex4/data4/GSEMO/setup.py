from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    name="complete_suite",
    ext_modules=cythonize(
        "complete_suite.pyx",
        language_level=3,
        annotate=False,  # Disable HTML annotation for faster compilation
        compiler_directives={
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True
        },
        nthreads=4  # Use 4 threads for compilation
    ),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)