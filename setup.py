from distutils.core import setup, Extension

setup(
    ext_modules=[
        Extension('utils', sources = ['src/find_peaks.cc']),
    ],
)
