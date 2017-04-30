# Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[calibration]: ./output_images/tests/undistorted.jpg "Undistorted"
[test_undist]: ./output_images/undistorted_test1.jpg "Road Transformed"
[sobel]: ./output_images/sobel_test4.jpg "Sobel Example"
[s_channel]: ./output_images/s_channel_test4.jpg "S Channel Example"
[combined]: ./output_images/combined_test4.jpg "Combined Example"
[src_points]: ./output_images/points/points_straight_lines2.jpg "Source points"
[dst_points]: ./output_images/points/dst_points_straight_lines2.jpg "Destination points"
[warped]: ./output_images/warped_straight_lines2.jpg "Warped Example"
[histogram]: ./output_images/histogram_test2.jpg "Hitogram Windows"
[sliding_windows]: ./output_images/lines_test2.jpg "Sliding Windows Example"
[lines]: ./output_images/output_test2.jpg "Lines Detected Example"
[lane]: ./output_images/result_test4.jpg "Lane Detected Example"
[video_output]: ./output.mp4 "Video Result"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in lines 38 through 48 of the file called [camera.py](./AdvLaneFinding/camera.py)).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `obj_p` is just a replicated array of coordinates, and `obj_points` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `img_points` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `obj_points` and `img_points` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![calibration]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![test_undist]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 30 through 34 in [P4.py](./P4.py) and implemented in lines 20 through 22, 44 through 53 and 103 through 109 in [threshold.py](./AdvLaenFinding/threshold.py)).  Here's some examples of my outputs for this step.
- Sobel

![sobel]

- S Channel

![s_channel]

- Combined

![combined]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in lines 42 through 55 in the file [transform.py](./AdvLaneFinding/transofrm.py).  The `warp()` function takes as inputs an image (`img`), as well as boolean value (`grayscale`) that indicates if the function needs to grayscales the image. I chose the hardcode the source and destination points in the following manner (see function `get_matrices()`, lines 32 through 40 in [transform.py](./AdvLaneFinding/transofrm.py)):

```python
src = np.float32([[595.0, 450.0], [259.0, 687.0], [1056.0, 687.0], [687.0, 450.0]])
dst = np.float32([[259.0, 0.0], [259.0, 720.0], [1056.0, 720.0], [1056.0, 0.0]])
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

- Source points

![src_points]

- Detination points

![dst_points]

- Warped

![warped]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial.

I used a histogramover the lower half of the image to find the peaks and, hence, the base of the two lane lines. Below, you can find the histogram superimposed to the wapre image resulting from the previous step. The function `histogram()`, in lines between 20 and 30 from [lines.py](./AdvLaneFinding/lines.py), returns this histogram.

![histogram]

Then, at the base of each line, a sliding window is used to find the pixels belonging to that lane line. This technique is used for both the left and the right lane lines and can be found at function `find_lines()` (lines 47 throigh 97 in [lines.py](./AdvLaneFinding/lines.py)). Example below.

![sliding_windows]


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculated the curvature radius in the function `find_lines()` (lines 147 throigh 153 in [lines.py](./AdvLaneFinding/lines.py)). Taking the detected left and right line pixels and multiplying them by a pixel to meter conversions and then fiting polynomials to the result points.

To find the car offset from the center I got the mean of the fits of both lane lines and substracted the half with of the image, then multplied the result by the pixel to meter conversion (line 155 in [lines.py](./AdvLaneFinding/lines.py)).

Example of the result below.

![lines]

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![lane]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

