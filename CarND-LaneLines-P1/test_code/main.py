#importing some useful packages
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import helper_func as helper

matplotlib.interactive(False)
plt.show(block=False)



#reading in an image
image = mpimg.imread('../test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)

#plt.figure(1)
#plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')


import os
print(os.listdir("../test_images/"))

# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.

# Grayscale the image
gray_im = helper.grayscale(image)
#plt.figure(2)
#plt.imshow(gray_im, cmap='gray')

# Define a kernel size and apply Gaussian smoothing
kernel_size = 5
blur_gray_im = helper.gaussian_blur(gray_im,kernel_size)

#plt.figure(3)
#plt.imshow(blur_gray_im, cmap='gray')

# Define parameters for Canny and apply
low_threshold = 50
high_threshold = 150
edges_im = helper.canny(blur_gray_im, low_threshold, high_threshold)
print('edges_im.shape='+str(edges_im.shape))

#plt.figure(4)
#plt.imshow(edges_im, cmap='gray')

# Create a masked edges_im using cv2.fillPoly()
mask = np.zeros_like(edges_im)
ignore_mask_color = 255

# Define a four sided polygon to mask
imshape = image.shape
print('imshape=' + str(imshape))

vertices = np.array([[[0,imshape[0]],[450,317],[490,317],[imshape[1],imshape[0]]]], dtype=np.int32)
masked_edges_im = helper.region_of_interest(edges_im, vertices)
print('masked_edges_im.shape='+str(masked_edges_im.shape))

#plt.figure(5)
#plt.imshow(masked_edges_im, cmap='gray')

# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 2 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 15 # minimum number of votes (intersections in Hough grid cell)
min_lin_length = 40 # minimum number of pixels making up a line
max_line_gap = 30 # maximum gap in pixels between connectable line segments
line_image = np.copy(image)*0

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
line_image = helper.hough_lines(masked_edges_im, rho, theta, threshold, min_lin_length, max_line_gap)
print('line_image.shape='+str(line_image.shape))
plt.figure(6)
plt.imshow(line_image)









# Create a "color" binary image to combine with line image
color_edges = np.dstack((line_image[:,:,2], line_image[:,:,2], line_image[:,:,2])) # three color channels
print('color_edges.shape='+str(color_edges.shape))

# Draw the lines on the edge image
lines_edges = helper.weighted_img(line_image, image, α=0.8, β=1., λ=0.)

#plt.figure(7)
#plt.imshow(lines_edges)




























plt.show()
#plt.close('all')