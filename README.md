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
[lane]: ./output_images/result_straight_lines1.jpg "Lane Detected Example"
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

I started by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `obj_p` is just a replicated array of coordinates, and `obj_points` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `img_points` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `obj_points` and `img_points` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![calibration]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![test_undist]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 23 through 65 in [P4.py](./P4.py) and implemented in lines 16 through 53 in [threshold.py](./AdvLaneFinding/threshold.py)). The color selection code is at lines 18 through 76 in [utils.py](./AdvLaneFinding/utils.py) .  Here's some examples of my outputs for this step.

- Sobel

![sobel]

- S Channel

![s_channel]

- Combined

![combined]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.
The code for my perspective transform includes a function called `warp()`, which appears in lines 42 through 55 in the file [transform.py](./AdvLaneFinding/transofrm.py).  The `warp()` function takes as inputs an image (`img`), as well as boolean value (`grayscale`) that indicates if the function needs to grayscales the image. I chose the hardcode the source and destination points in the following manner (see function `get_matrices()`, lines 32 through 40 in [transform.py](./AdvLaneFinding/transofrm.py)):

```python
src = np.float32([[1030, 670], [712, 468], [570, 468], [270, 670]])
dst = np.float32([[1010, 720], [1010, 0], [280, 0], [280, 720]])
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

- Source points

![src_points]

- Detination points

![dst_points]

- Warped

![warped]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial.

I used a histogram over the lower half of the image to find the peaks and, hence, the base of the two lane lines. Below, you can find the histogram superimposed to the warped image resulting from the previous step. The function `histogram()`, in lines between 24 and 34 from [lines.py](./AdvLaneFinding/lines.py), returns the next histogram.

![histogram]

Then, at the base of each line, a sliding window is used to find the pixels belonging to that lane line. This technique is used for both the left and the right lane lines and can be found at function `find_complete_line()` (lines 69 throigh 129 in [lines.py](./AdvLaneFinding/lines.py)). Example below.

![sliding_windows]


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculated the curvature radius in the function `get_line_curvature()` (lines 159 throigh 185 in [lines.py](./AdvLaneFinding/lines.py)). I taken the detected left and right line pixels and multiplyed them by a pixel-meter conversion and then fitted polynomials to the result points.

To find the car offset from the center I got the mean of the fits of both lane lines and subtracted the half of the image width, then multplied the result by the pixel-meter conversion (function `get_car_position()` at lines 187 through 194 in [lines.py](./AdvLaneFinding/lines.py)).

Example of the result below.

![lines]

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.
I implemented this step in lines 208 through 238 in my code in [lines.py](./AdvLaneFinding/lines.py) in the function `find_lines()`.  Here is an example of my result on a test image:

![lane]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).
Here's a [link to my video result](./output.mp4)
And here's a [link to the debug video result](./debug_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My approach is, somehow, a very mechanichal one. I mean, there is not "real intelligence" here, and it is quite slow. I doubt it can be reliable in a real environment.

To tune up all the relevant parameters I've chosen the try-fail technique. I'm aware that is not the most effective one (consumes lots of time) but, again, I'm learning, and I 'm enjoying each and every single test and error/correction that I've made.

Basically I've use the techniques showed in the courseware, complemented with some research of myself.

My pipeline fails incredibly with the `challenge` and `harder challenge` videos ... I think I will try later to improve my solution.

