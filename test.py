import random
from scipy.spatial.distance import cdist

import numpy as np

def dist(X, Y):
    return np.linalg.norm(X - Y, keepdims=False)


def find_cluster(X, center):
    D =cdist(X,center)
    return np.argmin(D,axis=1)

a = np.random.rand(6,2)
b = np.random.rand(3,2)
print(a)
print(b)
print(cdist(a,b))
print(find_cluster(a,b))
