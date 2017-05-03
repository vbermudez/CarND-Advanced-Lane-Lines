import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal

import AdvLaneFinding.utils as utils

class LineFinder(object):
    """
        A lane line finder utility.
    """

    def __init__(self):
        """
            Returns as LineFinder object.
        """
        self.n_frames = 7
        self.line_segments = 10
        self.image_offset = 250
        self.left_line = None
        self.right_line = None
        self.center_poly = None
        self.curvature = 0.0
        self.car_offset = 0.0
        self.dists = []
        self.ym_per_pix = 30 / 720  # meters per pixel in y dimension
        self.xm_per_pix = 3.7 / 700  # meteres per pixel in x dimension

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
    
    def detect_lane_poly(self, img, poly, steps):
        """
            Slides a window along a polynomial and select all pixels inside.
        """
        pixels_per_step = img.shape[0] // steps
        all_x = []
        all_y = []

        for i in range(steps):
            start = img.shape[0] - (i * pixels_per_step)
            end = start - pixels_per_step
            y_center = (start + end) // 2
            x_center = poly(y_center)
            half_size = pixels_per_step // 2
            window = img[y_center - half_size : y_center + half_size, x_center - half_size : \
                x_center + half_size]
            x, y = (window.T == 1).nonzero()
            all_x.extend(x + x_center - half_size)
            all_y.extend(y + y_center - half_size)
        return all_x, all_y

    def validate_lines(self, left_x, left_y, right_x, right_y):
        """
            Compares two line to each other and to their last prediction.
        """
        left_detected = False
        right_detected = False

        if self.is_line_plausible((left_x, left_y), (right_x, right_y)):
            left_detected = True
            right_detected = True
        elif self.left_line is not None and self.right_line is not None:
            if self.is_line_plausible((left_x, left_y), (self.left_line.ally, self.left_line.allx)):
                left_detected = True
            if self.is_line_plausible((right_x, right_y), \
                                                (self.right_line.ally, self.right_line.allx)):
                right_detected = True
        return left_detected, right_detected

    def is_line_plausible(self, left, right):
        """
            Determines if pixels describing two line are plausible
            lane lines based on curvature and distance.
        """
        if len(left[0]) < 3 or len(right[0]) < 3:
            return False
        else:
            dist_thresh = (350, 460)
            parallel_thresh = (0.0003, 0.55)
            new_left = Line(y=left[0], x=left[1])
            new_right = Line(y=right[0], x=right[1])
            is_parallel = new_left.get_current_is_parallel(new_right, threshold=parallel_thresh)
            dist = new_left.get_current_distance(new_right)
            is_plausible_dist = dist_thresh[0] < dist < dist_thresh[1]
            return is_parallel & is_plausible_dist

    def detect_line_hist(self, warped, steps, search_window, h_window):
        """
            Detects lane line pixels by applying a sliding windows over the histogram.
        """
        all_x = []
        all_y = []
        hist_thresh = 5
        num_peaks = 1
        masked = warped[:, search_window[0]:search_window[1]]
        pixels_per_step = warped.shape[0] // steps

        for i in range(steps):
            start = masked.shape[0] - (i * pixels_per_step)
            end = start - pixels_per_step
            histogram = np.sum(masked[end:start, :], axis=0)
            histogram_smooth = signal.medfilt(histogram, h_window)
            peaks = np.array(signal.find_peaks_cwt(histogram_smooth, np.arange(1, 5)))
            peaks = np.array(signal.find_peaks_cwt(histogram_smooth, np.arange(1, 5)))
            if len(peaks) == 0:
                highest_peak = []
            else:
                peak_list = [(peak, histogram_smooth[peak]) \
                            for peak in peaks if histogram_smooth[peak] > hist_thresh]
                peak_list = sorted(peak_list, key=lambda p: p[1], reverse=True)
                if len(peak_list) == 0:
                    highest_peak = []
                else:
                    px, py = zip(*peak_list)
                    px = list(px)
                    if len(peak_list) < num_peaks:
                        highest_peak = px
                    else:
                        highest_peak = px[:num_peaks]
            if len(highest_peak) == 1:
                highest_peak = highest_peak[0]
                center = (start + end) // 2
                half_size = pixels_per_step // 2
                window = masked[center - half_size:center + half_size, highest_peak - \
                        half_size:highest_peak + half_size]
                x, y = (window.T == 1).nonzero()
                all_x.extend(x + highest_peak - half_size)
                all_y.extend(y + center - half_size)
        all_x = np.array(all_x) + search_window[0]
        all_y = np.array(all_y)

        return all_x, all_y

    def get_curvature(self, fit):
        """
            Calculates the curvature of a line in meters
        """
        y = np.array(np.linspace(0, 719, num=10))
        x = np.array([fit(x) for x in y])
        y_eval = np.max(y)
        fit = np.polyfit(y * self.ym_per_pix, x * self.xm_per_pix, 2)
        return ((1 + (2 * fit[0] * y_eval + fit[1]) ** 2) ** 1.5) / np.absolute(2 * fit[0])

    def draw_line(self, img, poly, steps=5, color=255):
        """
            Draws a line.
        """
        img_height = img.shape[0]
        pixels_per_step = img_height // steps
        for i in range(steps):
            start = i * pixels_per_step
            end = start + pixels_per_step
            start_point = (int(poly(start)), start)
            end_point = (int(poly(end)), end)
            img = cv2.line(img, end_point, start_point, color, 10)
        return img

    def find_lines(self, warped, original, M, Minv, name='lines.jpg', output_dir='./output_images', draw=False):
        """
            Finds the lane lines. Code from lectures.
        """
        left_detected = False
        right_detected = False
        left_x = []
        left_y = []
        right_x = []
        right_y = []
        if self.left_line is not None and self.right_line is not None:
            left_x, left_y = self.detect_lane_poly(warped, \
                                self.left_line.best_fit_poly, self.line_segments)
            right_x, right_y = self.detect_lane_poly(warped, \
                                self.right_line.best_fit_poly, self.line_segments)
            left_detected, right_detected = self.validate_lines(left_x, left_y, right_x, right_y)
        if not left_detected:
            left_x, left_y = self.detect_line_hist(warped, self.line_segments, \
                                (self.image_offset, warped.shape[1] // 2), h_window=7)
            left_x, left_y = utils.remove_atypical(left_x, left_y)
        if not right_detected:
            right_x, right_y = self.detect_line_hist(warped, self.line_segments, \
                                (warped.shape[1] // 2, warped.shape[1] - self.image_offset), \
                                h_window=7)
            right_x, right_y = utils.remove_atypical(right_x, right_y)
        if not left_detected or not right_detected:
            left_detected, right_detected = self.validate_lines(left_x, left_y, right_x, right_y)
        if left_detected:
            if self.left_line is not None:
                self.left_line.update(y=left_x, x=left_y)
            else:
                self.left_line = Line(self.n_frames, left_y, left_x)
        if right_detected:
            if self.right_line is not None:
                self.right_line.update(y=right_x, x=right_y)
            else:
                self.right_line = Line(self.n_frames, right_y, right_x)
        if self.left_line is not None and self.right_line is not None:
            self.dists.append(self.left_line.get_best_fit_distance(self.right_line))
            self.center_poly = (self.left_line.best_fit_poly + self.right_line.best_fit_poly) / 2
            self.curvature = self.get_curvature(self.center_poly)
            self.car_offset = (warped.shape[1] / 2 - self.center_poly(719)) * self.xm_per_pix
            # render!
            overlay = np.zeros([original.shape[0], original.shape[1], original.shape[2]])
            mask = np.zeros([original.shape[0], original.shape[1]])
            steps = 20
            area_h = original.shape[0]
            points_left = np.zeros((steps + 1, 2))
            points_right = np.zeros((steps + 1, 2))
            for i in range(steps + 1):
                pixels_per_step = area_h // steps
                start = area_h - i * pixels_per_step
                points_left[i] = [self.left_line.best_fit_poly(start), start]
                points_right[i] = [self.right_line.best_fit_poly(start), start]
            lane_area = np.concatenate((points_left, points_right[::-1]), axis=0)
            mask = cv2.fillPoly(mask, np.int32([lane_area]), 1)
            mask = cv2.warpPerspective(mask, Minv, (mask.shape[1], mask.shape[0]))
            overlay[mask == 1] = (128, 255, 0)
            selection = (overlay != 0)
            original[selection] = original[selection] * 0.5 + overlay[selection] * 0.5
            mask[:] = 0
            mask = self.draw_line(mask, self.left_line.best_fit_poly)
            mask = self.draw_line(mask, self.right_line.best_fit_poly)
            mask = cv2.warpPerspective(mask, Minv, (mask.shape[1], mask.shape[0]))
            original[mask == 255] = (255, 200, 2)
            l_r = 'left' if self.car_offset < 0 else 'right'
            cv2.putText(original, \
                "Car is {:.2f}m {} of center".format(np.abs(self.car_offset), l_r), \
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(original, "Radius of curvature is {:.2f}m".format(self.curvature), \
                (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        return original

class Line:
    """
        Line holder object.
    """

    def __init__(self, n_frames=1, x=None, y=None):
        """
            Returns a line object.
        """
        self.n_frames = n_frames
        self.detected = False
        self.n_pixel_per_frame = []
        self.recent_xfitted = []
        self.bestx = None
        self.best_fit = None
        self.current_fit = None
        self.current_fit_poly = None
        self.best_fit_poly = None
        self.radius_of_curvature = None
        self.line_base_pos = None
        self.diffs = np.array([0, 0, 0], dtype='float')
        self.allx = None
        self.ally = None

        if x is not None:
            self.update(x, y)

    def update(self, x, y):
        """
            Updates the line representation.
        """
        self.allx = x
        self.ally = y
        self.n_pixel_per_frame.append(len(self.allx))
        self.recent_xfitted.extend(self.allx)
        if len(self.n_pixel_per_frame) > self.n_frames:
            n_x_to_remove = self.n_pixel_per_frame.pop(0)
            self.recent_xfitted = self.recent_xfitted[n_x_to_remove:]
        self.bestx = np.mean(self.recent_xfitted)
        self.current_fit = np.polyfit(self.allx, self.ally, 2)
        if self.best_fit is None:
            self.best_fit = self.current_fit
        else:
            self.best_fit = (self.best_fit * (self.n_frames - 1) + self.current_fit) / self.n_frames
        self.current_fit_poly = np.poly1d(self.current_fit)
        self.best_fit_poly = np.poly1d(self.best_fit)

    def get_current_is_parallel(self, other_line, threshold=(0, 0)):
        """
            Checks if two lines are parallel by comparing their first two coefficients.
        """
        first_coefi_dif = np.abs(self.current_fit[0] - other_line.current_fit[0])
        second_coefi_dif = np.abs(self.current_fit[1] - other_line.current_fit[1])
        is_parallel = first_coefi_dif < threshold[0] and second_coefi_dif < threshold[1]
        return is_parallel

    def get_current_distance(self, other_line):
        """
            Gets the distance between the current fit polynomials of two lines
        """
        return np.abs(self.current_fit_poly(719) - other_line.current_fit_poly(719))

    def get_best_fit_distance(self, other_line):
        """
            Gets the distance between the best fit polynomials of two lines
        """
        return np.abs(self.best_fit_poly(719) - other_line.best_fit_poly(719))
