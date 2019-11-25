from setuptools import setup, find_packages

REQUIRED_PACKAGES = ['torch>=1.3', 'numpy']

with open("README.md", "r") as h:
    long_description = h.read()

setup(
    name="mrphy",
    version="0.1.0",
    author="Tianrui Luo",
    author_email="tianrluo@umich.edu",
    description="A Pytorch based tool for MR physics simulations",
    license='MIT',
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/tianrluo/MRphy.py",
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)
