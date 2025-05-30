#!/usr/bin/env python3
"""
Setup script for CudaRL-Arena Python package.
"""

from setuptools import setup, find_packages

setup(
    name="cudarl",
    version="0.1.0",
    description="CUDA-accelerated Reinforcement Learning Arena",
    author="CudaRL-Arena Team",
    author_email="example@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)