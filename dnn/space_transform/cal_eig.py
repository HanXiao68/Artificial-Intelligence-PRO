#-*- encoding:utf-8 -*-
import sys

import numpy as np
from numpy import *
W=np.array([[1,1],[1,1]])
eigenvalue,featurevector=np.linalg.eig(W)

print("原始矩阵的特征值")
print("eigenvalue=",eigenvalue)
print("featurevector=",featurevector)
