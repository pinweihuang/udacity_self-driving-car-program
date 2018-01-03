import math
import cv2
import numpy as np

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    # for line in lines:
    #     for x1, y1, x2, y2 in line:
    #         cv2.line(img, (x1, y1), (x2, y2), color, thickness)


    # collect all line segements
    slopes = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slopes.append((y2-y1)/(x2-x1))
    slopes = np.array(slopes)

    print(slopes)

    sorted_slopes_idx = np.argsort(slopes)
    sorted_slopes = slopes[sorted_slopes_idx[:]]
    sorted_lines = lines[sorted_slopes_idx[:]]
    #
    # slopes_diff = sorted_slopes[1:] - sorted_slopes[:-1]
    # max_change_idx = np.argmax(slopes_diff)
    #
    # left_lines_slopes = sorted_slopes[:max_change_idx+1]
    # right_lines_slopes = sorted_slopes[max_change_idx+1:]
    #
    # left_lines = sorted_lines[:max_change_idx + 1]
    # right_lines = sorted_lines[max_change_idx + 1:]
    #
    # left_slope_avg = np.mean(left_lines_slopes)
    # right_slope_avg = np.mean(right_lines_slopes)
    #
    # left_lines_avg = np.mean(left_lines, axis=0)
    # right_lines_avg = np.mean(right_lines, axis=0)
    #
    # imshape = img.shape
    #
    #
    # x1_left = (imshape[0]-left_lines_avg[0][1])/left_slope_avg + left_lines_avg[0][0]
    # y1_left = imshape[0]
    #
    # x1_right = (imshape[0] - right_lines_avg[0][1]) / right_slope_avg + right_lines_avg[0][0]
    # y1_right = imshape[0]
    #
    # x2_left = (x1_left+x1_right)/2 - 20
    # x2_right = (x1_left + x1_right) / 2 + 20
    #
    # y2_left = y1_left + left_slope_avg*(x2_left-x1_left)
    # y2_right = y1_right + right_slope_avg * (x2_right - x1_right)
    #
    # cv2.line(img, (int(x1_left), int(y1_left)), (int(x2_left), int(y2_left)), color, thickness)
    # cv2.line(img, (int(x1_right), int(y1_right)), (int(x2_right), int(y2_right)), color, thickness)












def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)

    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.

def lane_finding_pipeline(img):
    # Grayscale the image
    gray_im = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # print('gray_im.shape='+str(gray_im.shape))

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray_im = gaussian_blur(gray_im, kernel_size)

    # Define parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    edges_im = canny(blur_gray_im, low_threshold, high_threshold)
    # print('edges_im.shape='+str(edges_im.shape))

    # Create a masked edges_im using cv2.fillPoly()
    mask = np.zeros_like(edges_im)
    ignore_mask_color = 255

    # Define a four sided polygon to mask
    imshape = img.shape
    # print('imshape=' + str(imshape))

    vertices = np.array([[[0, imshape[0]], [450, 317], [500, 317], [imshape[1], imshape[0]]]], dtype=np.int32)
    masked_edges_im = region_of_interest(edges_im, vertices)
    # print('masked_edges_im.shape='+str(masked_edges_im.shape))

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 30  # minimum number of votes (intersections in Hough grid cell)
    min_lin_length = 20  # minimum number of pixels making up a line
    max_line_gap = 100  # maximum gap in pixels between connectable line segments
    line_image = np.copy(image) * 0

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    line_image = hough_lines(masked_edges_im, rho, theta, threshold, min_lin_length, max_line_gap)
    print('line_image.shape=' + str(line_image.shape))

    # Create a "color" binary image to combine with line image
    color_edges = np.dstack((line_image[:, :, 2], line_image[:, :, 2], line_image[:, :, 2]))  # three color channels
    print('color_edges.shape=' + str(color_edges.shape))

    # Draw the lines on the edge image
    lines_edges = weighted_img(line_image, image, α=0.8, β=1., λ=0.)

    return lines_edges
