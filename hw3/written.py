# from sympy import *

# Beta, Sigma, Ita = Symbol('B'), Symbol('S'), Symbol('I')

# M = Matrix([[1-Beta, 2*Beta * Sigma **2], [Ita * (Beta-1), 1 - 2 * Ita* Beta * Sigma **2]])

# M.eigenvals()
import numpy as np
import scipy.linalg as sl
a = np.arange(25).reshape((5, 5))+1

# print(a)

kernal = np.array([[-1,-2,-1], [0,0,0],[1, 2, 1]])
# print(kernal)

def convolution2d(image, kernel, bias):
    m, n = kernel.shape
    if (m == n):
        y, x = image.shape
        y = y - m + 1
        x = x - m + 1
        new_image = np.zeros((y,x))
        for i in range(y):
            for j in range(x):
                new_image[i][j] = np.sum(image[i:i+m, j:j+m]*kernel) + bias
    return new_image

# print(convolution2d(a,kernal,0))

b = np.pad(a, 1)
print(b)
print(convolution2d(b,kernal,0))