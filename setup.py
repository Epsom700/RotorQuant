from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import os
import torch
os.environ['MACOSX_DEPLOYMENT_TARGET'] = '13.0'
torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')

setup(
    name='fwht_metal', 
    ext_modules=[
        CppExtension(
            name='fwht_metal',
            sources=['metal_kernel.mm'], 
            extra_compile_args={
                'cxx':[
                    '-ObjC++',
                    '-std=c++17',
                    '-arch', 'arm64',
                    '-mmacosx-version-min=13.0',
                    '-I/opt/homebrew/include',
                ]
            }, 
            extra_link_args = [
                '-framework', 'Metal',
                '-framework', 'Foundation',
                '-arch', 'arm64',
            ]
        )
    ], 
    cmdclass={'build_ext': BuildExtension}
)