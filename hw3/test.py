import numpy as np
a = np.zeros((4,5,1,3))
b = a[:, :, None, None]
print(a.shape)
print(b.shape)