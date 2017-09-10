from __future__ import print_function, division

import scipy.misc, numpy as np, os, sys
import random
import cv2
from threading import Thread
import datetime
from coral import coral_numpy # , coral_pytorch
# from color_transfer import color_transfer
import time


### Image helpers

def get_files(img_dir):
    files = os.listdir(img_dir)
    paths = []
    for x in files:
        paths.append(os.path.join(img_dir, x))
    # return [os.path.join(img_dir,x) for x in files]
    return paths

def save_img(out_path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(out_path, img)

def get_img(src):
   img = scipy.misc.imread(src, mode='RGB')
   if not (len(img.shape) == 3 and img.shape[2] == 3):
       img = np.dstack((img,img,img))
   return img

def center_crop(img, size=256):
    height, width = img.shape[0], img.shape[1]

    if height < size or width < size:  # Upscale to size if one side is too small
        img = resize_to(img, resize=size)
        height, width = img.shape[0], img.shape[1]

    h_off = (height - size) // 2
    w_off = (width - size) // 2
    return img[h_off:h_off+size,w_off:w_off+size]

def resize_to(img, resize=512):
    '''Resize short side to target size and preserve aspect ratio'''
    height, width = img.shape[0], img.shape[1]
    if height < width:
        ratio = height / resize
        long_side = round(width / ratio)
        resize_shape = (resize, long_side, 3)
    else:
        ratio = width / resize
        long_side = round(height / ratio)
        resize_shape = (long_side, resize, 3)
    
    return scipy.misc.imresize(img, resize_shape)

def get_img_crop(src, resize=512, crop=256):
    '''Get & resize image and center crop'''
    img = get_img(src)
    img = resize_to(img, resize)
    return center_crop(img, crop)

def get_img_random_crop(src, resize=512, crop=256):
    '''Get & resize image and random crop'''
    img = get_img(src)
    img = resize_to(img, resize=resize)
    
    offset_h = random.randint(0, (img.shape[0]-crop))
    offset_w = random.randint(0, (img.shape[1]-crop))
    
    img = img[offset_h:offset_h+crop, offset_w:offset_w+crop, :]

    return img

# def preserve_colors(content_rgb, styled_rgb):
#     """Extract luminance from styled image and apply colors from content"""
#     if content_rgb.shape != styled_rgb.shape:
#       new_shape = (content_rgb.shape[1], content_rgb.shape[0])
#       styled_rgb = cv2.resize(styled_rgb, new_shape)
#     styled_yuv = cv2.cvtColor(styled_rgb, cv2.COLOR_RGB2YUV)
#     Y_s, U_s, V_s = cv2.split(styled_yuv)
#     image_YUV = cv2.cvtColor(content_rgb, cv2.COLOR_RGB2YUV)
#     Y_i, U_i, V_i = cv2.split(image_YUV)
#     styled_rgb = cv2.cvtColor(np.stack([Y_s, U_i, V_i], axis=-1), cv2.COLOR_YUV2RGB)
#     return styled_rgb

def preserve_colors_np(style_rgb, content_rgb):
    coraled = coral_numpy(style_rgb/255., content_rgb/255.)
    coraled = np.uint8(np.clip(coraled, 0, 1) * 255.)
    return coraled

# def preserve_colors_pytorch(style_rgb, content_rgb):
#     coraled = coral_pytorch(style_rgb/255., content_rgb/255.)
#     coraled = np.uint8(np.clip(coraled, 0, 1) * 255.)
#     return coraled

# def preserve_colors_color_transfer(style_rgb, content_rgb):
#     style_bgr = cv2.cvtColor(style_rgb, cv2.COLOR_RGB2BGR)
#     content_bgr = cv2.cvtColor(content_rgb, cv2.COLOR_RGB2BGR)
#     transferred = color_transfer(content_bgr, style_bgr)
#     return cv2.cvtColor(transferred, cv2.COLOR_BGR2RGB)

### Video/Webcam helpers
### Borrowed from https://github.com/jrosebr1/imutils/

class WebcamVideoStream:
    '''From http://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/'''
    def __init__(self, src=0, width=None, height=None):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)

        if width is not None and height is not None: # Both are needed to change default dims
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        (self.ret, self.frame) = self.stream.read()

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self
 
    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
 
            # otherwise, read the next frame from the stream
            (self.ret, self.frame) = self.stream.read()
 
    def read(self):
        # return the frame most recently read
        return (self.ret, self.frame)
 
    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

class FPS:
        def __init__(self):
                # store the start time, end time, and total number of frames
                # that were examined between the start and end intervals
                self._start = None
                self._end = None
                self._numFrames = 0

        def start(self):
                # start the timer
                self._start = datetime.datetime.now()
                return self

        def stop(self):
                # stop the timer
                self._end = datetime.datetime.now()

        def update(self):
                # increment the total number of frames examined during the
                # start and end intervals
                self._numFrames += 1

        def elapsed(self):
                # return the total number of seconds between the start and
                # end interval
                return (self._end - self._start).total_seconds()

        def fps(self):
                # compute the (approximate) frames per second
                return self._numFrames / self.elapsed()
