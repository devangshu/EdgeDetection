import numpy as np
import matplotlib.pyplot as plt

vertical_filter = np.matrix([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]])

horizontal_filter = vertical_filter.transpose()

image = plt.imread('longhorn.png')


print(image.shape)
n,m,d = image.shape

edges = image.copy()

countIterations = 0
for row in range(3, n-2):
  for col in range(3, m-2):
    countIterations += 1
    local_pixels = image[row-1:row+2, col-1:col+2, 0]

    vertical_tranformed_pixels = vertical_filter*local_pixels


    vertical_score = vertical_tranformed_pixels.sum()/4

    horizontal_transformed_pixels = horizontal_filter*local_pixels

    horizontal_score = horizontal_transformed_pixels.sum()/4

    edge_score = np.sqrt(np.power(vertical_score, 2) + np.power(horizontal_score, 2))
    edges[row,col] = [edge_score]*4

    if countIterations == 1:
      print("The local pixel value is:")
      print(local_pixels)

      print("The vertical transformed pixels are:")
      print(vertical_tranformed_pixels)
      print("The vertical score is:")
      print(vertical_score)
      
      print("The horizontal transformed pixels are:")
      print(horizontal_transformed_pixels)
      print("The horizontal score is:")
      print(horizontal_score)

      print("The edge score is:")
      print(edge_score)



edges =  edges/edges.max()

plt.title('Original Image')
plt.imshow(image)
plt.show()

plt.title('Modified Image')
plt.imshow(edges)
plt.show()

"""
def computeVerticalScore( kernel_to_convolve):
    return ((kernel_to_convolve*vertical_filter).sum()/4)

def computerHorizontalScore( kernel_to_convolve ):
    return ((kernel_to_convolve*horizontal_filter).sum()/4)

def sobel:
    for row in range(3, n-2):
        for col in range(3, m-2):
            pass
            """
