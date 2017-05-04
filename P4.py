import os
import numpy as np
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

    def process(self, img, name='output.jpg', draw=False):
        """
            Processes an image.
        """
        # gblur = utils.gaussian_blur(img)
        # undist = self.trans.undistort(gblur)
        undist = self.trans.undistort(img)
        # sobel = self.thresh.sobel(undist)
        # hls = utils.rgb2hls(undist)
        # s_channel = utils.select_channel(hls)
        # s_channel_thresh = self.thresh.threshold(s_channel, thresh=(170, 255))
        # combined = self.thresh.combine_two(s_channel_thresh, sobel)

        mag = self.thresh.grad_magnitude(undist, 5, (130, 255))
        equ = utils.equalize(undist)
        color_thresh = self.thresh.threshold(equ, (251, 255))
        combined = self.thresh.combine_two(mag, color_thresh)

        # r_channel = utils.select_channel(undist, 'rgb', 'r')
        # r_channel_thresh = self.thresh.threshold(r_channel, thresh=(170, 255))
        # luv = utils.rgb2luv(undist)
        # l_channel = utils.select_channel(luv, 'luv', 'l')
        # l_channel_thresh = self.thresh.threshold(l_channel, thresh=(170, 255))
        # lab = utils.rgb2lab(undist)
        # b_channel = utils.select_channel(lab, 'lab', 'l')
        # b_channel_thresh = self.thresh.threshold(b_channel, thresh=(170, 255))
        # l_channel = utils.select_channel(hls, 'l')
        # l_channel_thresh = self.thresh.threshold(l_channel, thresh=(170, 255))
        
        # gray_thrsh = self.thresh.gray_threshold(undist)
        # white_yellow = utils.extract_white_yellow(undist)
        # combined = np.zeros_like(s_channel_thresh)
        # combined[(white_yellow == 255) | ((s_channel_thresh == 1) & \
        #     (l_channel_thresh == 1)) | ((sobel == 1) & (gray_thrsh == 1)) | \
        #     (l_channel_thresh == 1)] = 1
        
        
        # region = utils.get_region(combined)
        # warped = self.trans.warp(region)
        warped = self.trans.warp(combined)
        # result = self.lines.find_lines(warped, img, name, draw)
        result = self.lines.process_image_ex(warped, img, combined, name, draw)
        return result

def pipeline():
    """
        Detects the lines on video imput.
    """
    cal = Calibrator()
    ret, mtx, dist, rvecs, tvecs = cal.calibrate()
    trans = Transformer(mtx, dist)
    thresh = Thresholder()
    lines = LineFinder(trans)
    proc = Processor(thresh, trans, lines)
    video = VideoFileClip('./project_video.mp4')
    output = video.fl_image(proc.process)
    output.write_videofile('./output.mp4', audio=False)

def test_pipeline():
    """
        Tests the pipelin with static images
    """
    cal = Calibrator()
    ret, mtx, dist, rvecs, tvecs = cal.calibrate()
    trans = Transformer(mtx, dist)
    thresh = Thresholder()
    lines = LineFinder(trans)
    proc = Processor(thresh, trans, lines)
    path = './test_images'
    out_path = './output_images'
    for img_name in utils.list_dir(path):
        base_path, name = os.path.split(img_name)
        print('Processing ' + name + '...')
        img = utils.read_image(img_name)
        result = proc.process(img, 'output_' + name, True)
        utils.write_image(result, os.path.join(out_path, 'result_' + name))

if __name__ == "__main__":
    # test_pipeline()
    pipeline()

