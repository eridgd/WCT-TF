import cv2
import datetime
from threading import Thread


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
