from distutils.core import setup
from Cython.Build import cythonize

import numpy as np

setup(
    name="exponential_levenshtein",
    ext_modules=cythonize("/home/lsz/MarkText/watermark/KGW_watermark/exponential/exponential_levenshtein.pyx"),
    include_dirs=[np.get_include()]
)
print(np.get_include())
