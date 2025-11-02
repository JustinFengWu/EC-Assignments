from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name="ex3_fast",
    ext_modules=cythonize("ex3_fast.pyx", language_level=3, annotate=True),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)
