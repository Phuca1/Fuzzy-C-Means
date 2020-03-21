import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score

np.random.seed(10)
mean = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 500

X0 = np.random.multivariate_normal(mean[0], cov, N)
X1 = np.random.multivariate_normal(mean[1], cov, N)
X2 = np.random.multivariate_normal(mean[2], cov, N)

X = np.concatenate((X0, X1, X2), axis=0)
K = 3
m = 2.00
ori_label = np.asarray([0] * N + [1] * N + [2] * N).T


def display(X, label):
    X0 = X[label == 0, :]
    X1 = X[label == 1, :]
    X2 = X[label == 2, :]

    plt.plot(X0[:, 0], X0[:, 1], 'bd', markersize=4)
    plt.plot(X1[:, 0], X1[:, 1], 'rd', markersize=4)
    plt.plot(X2[:, 0], X2[:, 1], 'yd', markersize=4)

    plt.axis('equal')
    plt.show()
N =X.shape[0]
display(X,ori_label)

def dist(A, B):
    return np.linalg.norm(A - B, keepdims=False)


# def init_center(X,K):
#     return X[np.random.choice(X.shape[0],K, replace=False)]

def init_membership(k, n):
    mem = []
    for i in range(n):
        num = np.random.rand(k)
        sums = sum(num)
        tmp_list = num / sums
        mem.append(tmp_list)
    mems = np.asarray(mem)
    return mems.T


def update_center(X, mem, m, K, N):
    v = []
    for i in range(K):
        num = np.zeros((1, X.shape[1]))
        den = 0
        for k in range(N):
            x = (mem[i][k] ** m) * X[k, :]
            num = num + x
            den += (mem[i][k] ** m)
            vx = num / den
        v.append(vx)
    va = np.asarray(v)
    # va.reshape(K*2)
    # va.reshape(K,2)
    return va


def update_mem(X, center, m, K, N):
    mem = []
    p = 2 / (m - 1)
    for j in range(K):
        arr = []
        for i in range(N):
            sums = 0
            for k in range(K):
                t = dist(X[i], center[j]) / dist(X[i], center[k])
                q = t ** p
                sums += q
                uik = 1 / sums
            arr.append(uik)
        mem.append(arr)

    u = np.asarray(mem)
    return u


def find_cluster(X, center):
    D = cdist(X, center)
    return np.argmin(D,axis=1)


def fuzzy_Cmeans(X, K, m, eps=1e-4, max_count=1e+5):

    N = X.shape[0]
    mem = init_membership(K, N)
    count = 0
    while True:
        cen = update_center(X, mem, m, K, N)
        c = cen.reshape(K * 2)
        cen = c.reshape(K, 2)
        tmp = mem
        mem = update_mem(X, cen, m, K, N)
        label = find_cluster(X, cen)
        count += 1
        if np.linalg.norm(mem-tmp)<eps or count >= max_count:
            break
    return (cen,label)

centers , labels = fuzzy_Cmeans(X,K,m)
print(centers)
display(X,labels)

# mem = init_membership(K,N)
# cen = update_center(X,mem,m,K,N)
#
#
# tmp = update_mem(X,cen,m,K,N)
# label = find_cluster(X,cen)
#
# print(X)
# print(mem)
# print(cen)
# print(tmp)
# print(label)
# print(np.sum(tmp[:,0]))

