#!/usr/bin/env python
# coding: utf-8

"""
Benchmark of 1000x1000x10 tensor operations.

1. Pure Python
2. Numpy
3. Numba
3. PyTorch
"""

def purepython():
    for i in range(1024):
        for j in range(1024):
            for t in range(10):
                pass
