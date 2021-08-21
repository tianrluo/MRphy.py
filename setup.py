from setuptools import setup, find_packages
import ctypes


def cuda_is_available():
    libnames = ('libcuda.so', 'libcuda.dylib', 'cuda.dll')
    for name in libnames:
        try:
            ctypes.CDLL(name)
        except OSError:
            continue
        else:
            return True
    else:
        return False
    return False


REQUIRED_PACKAGES = ['torch>=1.3', 'numpy', 'scipy', 'cupy>=7.0.0']
if not cuda_is_available():
    REQUIRED_PACKAGES.remove('cupy>=7.0.0')

with open("README.md", "r") as h:
    long_description = h.read()

setup(
    name="mrphy",
    version="0.1.10",
    author="Tianrui Luo",
    author_email="tianrluo@umich.edu",
    description="A Pytorch based tool for MR physics simulations",
    license='MIT',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tianrluo/MRphy.py",
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
