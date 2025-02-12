from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="attention_cuda",
    ext_modules=[
        CUDAExtension(
            "attention_cuda",
            ["attention_cuda.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)
