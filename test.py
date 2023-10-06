import numpy as np
a = (np.array([1,2,3]), np.array([4,5,6]))

for row, col in zip(a[0], a[1]):
    print(row, col)
