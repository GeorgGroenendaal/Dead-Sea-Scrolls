# %%
import numpy as np

from scipy import ndimage

a = np.array([[1, 0, 0], [1, 0, 1], [1, 0, 0], [1, 0, 1]])
print(a)


components, n_components = ndimage.label(a)

print(components)


heights = []
for i in range(1, n_components + 1):
    x = components == i
    vertical_sum = x.sum(axis=0)
    heights.append(vertical_sum.max())


print(heights)
