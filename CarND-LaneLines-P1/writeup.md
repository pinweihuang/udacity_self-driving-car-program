# **Finding Lane Lines on the Road** 


---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:

* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./test_images_output/gray.jpg "Grayscale"
[image2]: ./test_images_output/solidYellowCurve2.jpg

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My piple line consists of 6 steps:

* Convert the image to grayscale image
* Blur the image with `cv2.GaussianBlur()`
* Identify edges by `cv2.Canny()`
* Create a masked area for line detection
* Use `cv2.HoughLinesP()` to correlct line segements
* Implement the `draw_lines()` function to draw lines on the raw image

For `draw_lines()` function's modification, I first remove thoses line segements with unreasonable slope values (e.g. slope < 0.2), and categorize the collected slopes into **left group** and **right group**. Take the **average** values of the slope for **left slopes** and **right slopes** respectively, as well as **left points** and **right points** respectively. The program now has 1 **slope (m)** and 1 **point (x0,y0)** for drawing a **line**. What comes next is to extrapolate the line to the bottom of the masked image, forming a second point **(x1,y1)**. To draw a complete line, the program once again extrapolate the line to the top of the masked image, forming a third point **(x2,y2)**. With (x1,y1) and (x2,y2), use c`v2.line()` to generate a line image. There are chances that there is no element in left group or right group, where I plot no line to the image.


![Grayscale][image1]
![line][image2]



### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when both left line segements and righ line segements are not identified, in the cases of very blurred edges or very curved edges.

### 3. Suggest possible improvements to your pipeline

A possible improvement would be to use a polynomial fitting with the collected line segaments instead of straight line fitting.
