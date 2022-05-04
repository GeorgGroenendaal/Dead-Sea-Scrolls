import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

np.random.seed(1)
n = 10
l = 256
# //im = np.zeros((l, l))
# //points = l*np.random.random((2, n**2))
# # im[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
# # im = ndimage.gaussian_filter(im, sigma=l/(4.*n))

# mask = (im > im.mean()).astype(np.float)
# mask += 0.1 * im

img = plt.imread("C:/Users/mw0121921/Downloads/image-data/image-data/P123-Fg002-R-C01-R01-binarized.jpg");

hist, bin_edges = np.histogram(img, bins="auto")
bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])

horizontal_hist = img.shape[1] - np.sum(img,axis=1,keepdims=True)/255

#binary_img = img > 0.5

plt.figure(figsize=(11,4))

plt.subplot(131)
plt.imshow(img)
plt.axis('off')
plt.subplot(132)
#plt.plot(bin_centers, hist)
plt.hist(img,bin="auto")
plt.axvline(0.5, color='r', ls='--')
plt.text(0.57, 0.8, 'histogram', fontsize=20, transform = plt.gca().transAxes)
plt.yticks([])
plt.subplot(133)
plt.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
plt.axis('off')

plt.subplots_adjust(wspace=0.02, hspace=0.3, top=1, bottom=0.1, left=0, right=1)
plt.show()



