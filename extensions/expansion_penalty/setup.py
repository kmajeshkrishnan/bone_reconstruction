from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='expansion_penalty',
    ext_modules=[
        CUDAExtension('expansion_penalty', [
            'expansion_penalty.cpp',
            'expansion_penalty_cuda.cu',
        ],
        extra_compile_args={
            'cxx': [],
            'nvcc': ['-allow-unsupported-compiler'],
        }),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })