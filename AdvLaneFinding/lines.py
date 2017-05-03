import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks_cwt

import AdvLaneFinding.utils as utils

class LineFinder(object):
    """
        A lane line finder utility.
    """

    def __init__(self, transform):
        """
            Returns as LineFinder object.
        """
        self.left_line = Line()
        self.right_line = Line()
        self.ym_per_pix = 30/720 #image.shape[1] # meters per pixel in y dimension
        self.xm_per_pix = 3.7/700 # meteres per pixel in x dimension
        self.transform = transform

    def histogram(self, img):
        """
            Returns the histogram of an image.

            Attributes:
                img: the image.
                plot_hist: specifies if the histogram must be plotted.
                output: where to write the plotted histogram.
                name: name of the plotted histogram.
        """
        return np.sum(img[img.shape[0] / 2:, :], axis=0)

    def find_lines_base(self, img):
        """
            Finds the coordinates of the base of the lane line (left, right)
        """
        histogram = self.histogram(img)
        indexes = find_peaks_cwt(histogram, np.arange(1, 550))
        return [(indexes[0], img.shape[0]), (indexes[-1], img.shape[0])]

    def find_complete_line(self, img, lane_base):
        """
            Uses a window of size 100px to find all the pixels in that lane.
        """
        window_size = 100 * 2
        x_base = lane_base[0]
        if x_base > window_size:
            window_low = x_base - window_size / 2
        else:
            window_low = 0
        window_high = x_base + window_size / 2
        window = img[:, window_low : window_high]
        x, y = np.where(window == 1)
        y += np.uint64(window_low)
        return (x, y)

    def get_curved_line(self, px):
        """
            Calculates a 2nd order polynomial that fits the pixels
        """
        x, y = px
        degree = 2
        return np.polyfit(x, y, deg=degree)

    def get_line_curvature(self, img, px):
        """
            Calculates the curvature of a line.
        """
        y, x = px
        y_eval = np.max(y)
        fit = np.polyfit(y * self.ym_per_pix, x * self.xm_per_pix, 2)
        return int(((1 + (2 * fit[0] * y_eval + fit[1])**2)**1.5) / np.absolute(2 * fit[0]))

    def get_car_position(self, img, ll_base, rl_base):
        """
            Calculates the distance of the car from the center of the lane.
        """
        image_center = (img.shape[1]/2, img.shape[0])
        car_middle_pixel = int((ll_base[0] + rl_base[0])/2)
        return (car_middle_pixel - image_center[0]) * self.xm_per_pix

    def draw_curved_line(self, img, line, color=(255, 0, 0), thickness=50):
        """
            Draws a curved line.
        """
        p = np.poly1d(line)
        x = list(range(0, img.shape[0]))
        y = list(map(int, p(x)))
        pts = np.array([[_y, _x] for _x, _y in zip(x, y)])
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, np.int32([pts]), False, color=color, thickness=thickness)
        return pts

    def draw_lines(self, img, ll_pixels, rl_pixels, ll_base, rl_base):
        """
            Draws the lane lines on the image
        """
        output = np.zeros_like(img)
        line1 = self.get_curved_line(ll_pixels)
        line1_pts = self.draw_curved_line(output, line1)
        left_line_curvature = self.get_line_curvature(output, ll_pixels)
        line2 = self.get_curved_line(rl_pixels)
        line2_pts = self.draw_curved_line(output, line2, color=(0, 0, 255))
        right_line_curvature = self.get_line_curvature(output, rl_pixels)
        top_points = [line1_pts[-1], line2_pts[-1]]
        base_points = [line1_pts[0], line2_pts[0]]
        distance_from_left = self.get_car_position(output, ll_base, rl_base)
        cv2.fillPoly(output, [np.concatenate((line2_pts, line1_pts, top_points, base_points))], \
            color=(0, 255, 0))
        return (output, left_line_curvature, right_line_curvature, distance_from_left)

    def find_lines(self, warped, original, name='lines.jpg', output_dir='./output_images', draw=False):
        """
            Finds the lane lines. Code from lectures.

            Attributes:
                img: the image.
        """
        try:
            lines_base = self.find_lines_base(warped)
            left_line_base, right_line_base = lines_base
            left_line_pixels = self.find_complete_line(warped, left_line_base)
            right_line_pixels = self.find_complete_line(warped, right_line_base)
            warped_w_lines, left_curv, right_curv, car_offset = self.draw_lines(original, \
                left_line_pixels, right_line_pixels, left_line_base, right_line_base)
            lane_lines = self.transform.unwarp(warped_w_lines)
            output = cv2.addWeighted(original, 1, lane_lines, .3, 0)
            cv2.putText(output, "Car offset: %.2fm" % car_offset, (10, 50), \
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(output, "Left curvature: %.2fm" % left_curv, (10, 80), \
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(output, "Right curvature: %.2fm" % right_curv, (10, 110), \
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        except Exception as e:
            print(e)
            return original
        return output

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
