%matplotlib inline

import math
import time
import numpy as np
import torch
from d2l import torch as d2l

n = 10000
a = torch.ones(n)
b = torch.ones(n)
c = torch zeros(n)
t = time.time()
# Add 1 coordinate at a time using a for loop. This is slow and represents how the processor is much faster
# at doing calculations than moving data from main memory to CPU cache
# for i in range(n):
#     c[i] = a[i] + b[i]
# f'{time.time()-t:.5f}sec'


# Compute elementwise sum which is a much faster method than the method commented above
d = a+b
f'{time.time()-t: .5f}sec'
