import os
from moviepy.editor import VideoFileClip

from AdvLaneFinding.camera import Calibrator
from AdvLaneFinding.transform import Transformer
from AdvLaneFinding.threshold import Thresholder
from AdvLaneFinding.lines import LineFinder
import AdvLaneFinding.utils as utils

class Processor(object):
    """
        Image processor utility.
    """
    def __init__(self, thresh, trans, lines):
        """
            Returns an image processor utility.
        """
        self.thresh = thresh
        self.trans = trans
        self.lines = lines
        M, Minv = self.trans.get_matrices()
        self.M = M
        self.Minv = Minv

    def process(self, img):
        """
            Processes an image.
        """
        undist = self.trans.undistort(img)
        sobel = self.thresh.sobel(undist)
        hls = utils.rgb2hls(undist)
        s_channel = utils.select_channel(hls)
        s_channel_thresh = self.thresh.threshold(s_channel, thresh=(170, 255))
        combined = self.thresh.combine_two(s_channel_thresh, sobel)
        warped = self.trans.warp(combined)
        out_img, result = self.lines.find_lines(warped, undist, self.M, self.Minv)
        return result

def pipeline():
    """
        Detects the lines on video imput.
    """
    cal = Calibrator()
    ret, mtx, dist, rvecs, tvecs = cal.calibrate()
    trans = Transformer(mtx, dist)
    thresh = Thresholder()
    lines = LineFinder()
    proc = Processor(thresh, trans, lines)
    video = VideoFileClip('./project_video.mp4')
    output = video.fl_image(proc.process)
    output.write_videofile('./output.mp4', audio=False)

if __name__ == "__main__":
    pipeline()

