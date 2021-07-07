import numpy as np
import matplotlib.pyplot as plt

vertical_filter = np.matrix([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]])

horizontal_filter = vertical_filter.transpose()

image = plt.imread('red_cross.png')

plt.title('Original Image')
plt.imshow(image)
plt.show()

n,m,d = image.shape

edges_image = image.copy()

def computeVerticalScore( kernel_to_convolve):
    return ((kernel_to_convolve*vertical_filter).sum()/4)

def computerHorizontalScore( kernel_to_convolve ):
    return ((kernel_to_convolve*horizontal_filter).sum()/4)

def sobel:
    for row in range(3, n-2):
        for col in range(3, m-2):

