import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

import AdvLaneFinding.utils as utils

class LineFinder(object):
    """
        A lane line finder utility.
    """

    def __init__(self):
        """
            Returns as LineFinder object.
        """
        self.left_line = Line()
        self.right_line = Line()

    def histogram(self, img):
        """
            Returns the histogram of an image.

            Attributes:
                img: the image.
                plot_hist: specifies if the histogram must be plotted.
                output: where to write the plotted histogram.
                name: name of the plotted histogram.
        """
        return np.sum(img[img.shape[0] // 2:, :], axis=0)

    def find_lines(self, warped, undistorted, M, Minv, name='lines.jpg', output_dir='./output_images', draw=False):
        """
            Finds the lane lines. Code from lectures.

            Attributes:
                img: the image.
        """
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Set the width of the windows +/- margin
        margin = 100
        # Create an output image to draw on and  visualize the result
        output = np.dstack((warped, warped, warped)) * 255
        if not self.left_line.detected:
            # Take a histogram of the bottom half of the image
            hist = self.histogram(warped)
            # Find the peak of the left and right halves of the histogram
            # These will be the starting point for the left and right lines
            midpoint = np.int(hist.shape[0]/2)
            leftx_base = np.argmax(hist[:midpoint])
            rightx_base = np.argmax(hist[midpoint:]) + midpoint
            # Choose the number of sliding windows
            nwindows = 9
            # Set height of windows
            win_h = np.int(warped.shape[0] / nwindows)
            # Current positions to be updated for each window
            leftx_current = leftx_base
            rightx_current = rightx_base
            # Set minimum number of pixels found to recenter window
            minpix = 50
            # Create empty lists to receive left and right lane pixel indices
            left_lane_inds = []
            right_lane_inds = []
            # Step through the windows one by one
            for window in range(nwindows):
                # Identify window boundaries in x and y (and right and left)
                win_y_low = warped.shape[0] - (window + 1) * win_h
                win_y_high = warped.shape[0] - window*win_h
                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin
                if draw:
                    # Draw the windows on the visualization image
                    cv2.rectangle(output, (win_xleft_low, win_y_low), \
                        (win_xleft_high, win_y_high), (0, 255, 0), 2)
                    cv2.rectangle(output, (win_xright_low, win_y_low), \
                        (win_xright_high, win_y_high), (0, 255, 0), 2)
                # Identify the nonzero pixels in x and y within the window
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & \
                    (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & \
                    (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
                # Append these indices to the lists
                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)
                # If you found > minpix pixels, recenter next window on their mean position
                if len(good_left_inds) > minpix:
                    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                if len(good_right_inds) > minpix:
                    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            # Concatenate the arrays of indices
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        else:
            # use areas around last fitted lines to find line pixels
            left_lane_inds = ((nonzerox > (self.left_line.current_fit[0] * (nonzeroy**2) + \
                self.left_line.current_fit[1] * nonzeroy + \
                self.left_line.current_fit[2] - margin)) & \
                (nonzerox < (self.left_line.current_fit[0]*(nonzeroy**2) + \
                self.left_line.current_fit[1] * nonzeroy + \
                self.left_line.current_fit[2] + margin)))
            right_lane_inds = ((nonzerox > (self.right_line.current_fit[0] * (nonzeroy**2) + \
                self.right_line.current_fit[1] * nonzeroy + \
                self.right_line.current_fit[2] - margin)) & \
                (nonzerox < (self.right_line.current_fit[0] * (nonzeroy**2) + \
                self.right_line.current_fit[1] * nonzeroy + \
                self.right_line.current_fit[2] + margin)))

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
        output[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        output[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        if draw:
            plt.imshow(output)
            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='yellow')
            plt.xlim(0, 1280)
            plt.ylim(720, 0)
            plt.savefig(os.path.join(output_dir, name), bbox_inches='tight')
            plt.close('all')
        y_eval = np.max(ploty)
        left_curverad = ((1 + (2 * left_fit[0] * y_eval + \
            left_fit[1])**2)**1.5) / np.absolute(2 * left_fit[0])
        right_curverad = ((1 + (2 * right_fit[0] * y_eval + \
            right_fit[1])**2)**1.5) / np.absolute(2 * right_fit[0])
        self.left_line.was_detected(left_fitx, left_curverad, left_fit, right_curverad, right_fit)
        self.right_line.was_detected(right_fitx, right_curverad, right_fit, left_curverad, \
            left_fit, not self.left_line.detected)
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
         # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + \
            left_fit_cr[1])**2)**1.5) / np.absolute(2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + \
            right_fit_cr[1])**2)**1.5) / np.absolute(2 * right_fit_cr[0])
        # car offset from center
        car_offset = ((left_fit[2] + right_fit[2]) / 2.0 - warped.shape[0] / 2.0) * xm_per_pix
        self.left_line.set_output_params(left_curverad, car_offset)
        self.right_line.set_output_params(right_curverad, car_offset)
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
            # Recast the x and y points into usable format for cv2.fillPoly()
        left_avg_fitx = self.left_line.best_fit[0] * ploty**2 + self.left_line.best_fit[1] * \
            ploty + self.left_line.best_fit[2]
        right_avg_fitx = self.right_line.best_fit[0] * ploty**2 + self.right_line.best_fit[1] * \
            ploty + self.right_line.best_fit[2]
        pts_left = np.array([np.transpose(np.vstack([left_avg_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_avg_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        # Draw the lane onto the warped blank image
        cv2.polylines(color_warp, np.int32([pts_left]), False, (255, 0, 0), 50)
        cv2.polylines(color_warp, np.int32([pts_right]), False, (0, 0, 255), 50)
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (color_warp.shape[1], color_warp.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)
        cv2.putText(result, "Car offset: " + str(self.left_line.mean_car_offset), (10, 50), \
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(result, "Left curvature: " + str(self.left_line.mean_curvature), (10, 80), \
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(result, "Right curvature: " + str(self.right_line.mean_curvature), (10, 110), \
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        return output, result

class Line(object):
    """
        Line holder object.
    """

    def __init__(self):
        """
            Returns a line object.
        """
        self.detected = False
        self.recent_xfitted = []
        self.bestx = None
        self.recent_fit = []
        self.best_fit = None
        self.current_fit = np.array([0, 0, 0], dtype='float')
        self.radius_of_curvature = None
        self.line_base_pos = None
        self.diffs = np.array([0, 0, 0], dtype='float')
        self.allx = None
        self.ally = None
        self.last_fit_suspitious = False
        self.recent_curvatures = []
        self.mean_curvature = None
        self.recent_car_offsets = []
        self.mean_car_offset = None
        self.output_frames = 0

    def was_detected(self, next_x, next_curvature, next_fit, next_other_curvature, next_other_fit, other_line_not_detected=False):
        """
            Evaluates if the line has been detected.
        """
        prev_detected = self.detected
        this_detected = self.best_fit is None or \
            ((np.abs(self.radius_of_curvature - next_curvature) < 5000 \
            or (self.radius_of_curvature > 5000 and next_curvature > 5000)) and \
            (np.abs(self.current_fit - next_fit) < [0.005, 2.0, 150.0]).all() and \
            (np.abs(next_other_curvature - next_curvature) < 5000  or \
            (next_other_curvature > 5000 and next_curvature > 5000)) and \
            (np.abs(next_other_fit[0] - next_fit[0]) < 0.001) and \
            (np.abs(next_other_fit[1] - next_fit[1]) < 0.5))
        self.detected = not prev_detected or not other_line_not_detected and this_detected

        if self.detected:
            if len(self.recent_xfitted) >= 4:
                self.recent_xfitted.pop(0)
            self.recent_xfitted.append(next_x)
            self.bestx = np.mean(self.recent_xfitted, axis=0)
            if len(self.recent_fit) >= 4 and not self.last_fit_suspitious:
                self.recent_fit.pop(0)
            if self.last_fit_suspitious:
                self.recent_fit.pop()
            self.last_fit_suspitious = not this_detected
            self.recent_fit.append(next_fit)
            self.best_fit = np.mean(self.recent_fit, axis=0)
            self.current_fit = next_fit
            self.radius_of_curvature = next_curvature

    def set_output_params(self, curvature, car_offset):
        """
            Sets the output paramters.
        """
        if self.detected:
            if len(self.recent_curvatures) >= 4 and not self.last_fit_suspitious:
                self.recent_curvatures.pop(0)
                self.recent_car_offsets.pop(0)
            if self.last_fit_suspitious:
                self.recent_car_offsets.pop()
                self.recent_curvatures.pop()
            self.recent_curvatures.append(curvature)
            self.recent_car_offsets.append(car_offset)
            if self.output_frames == 0:
                self.mean_curvature = np.mean(self.recent_curvatures)
                self.mean_car_offset = np.mean(self.recent_car_offsets)
            if self.output_frames >= 4:
                self.output_frames = 0
            else:
                self.output_frames += 1   