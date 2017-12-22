from __future__ import print_function, division

import os
import argparse
import cv2
import time
import numpy as np
import tensorflow as tf
from utils import preserve_colors_np
from utils import get_files, get_img, get_img_crop, resize_to, center_crop, save_img
from webcam_utils import WebcamVideoStream, FPS
from wct import WCT


parser = argparse.ArgumentParser()
parser.add_argument('-src', '--source', dest='video_source', type=int,
                    default=0, help='Device index of the camera.')
parser.add_argument('--checkpoints', nargs='+', type=str, help='List of checkpoint directories', required=True)
parser.add_argument('--relu-targets', nargs='+', type=str, help='List of reluX_1 layers, corresponding to --checkpoints', required=True)
parser.add_argument('--style-path', type=str, dest='style_path', help='Style images folder', required=True)
parser.add_argument('--vgg-path', type=str,
                    dest='vgg_path', help='Path to vgg_normalised.t7', 
                    default='models/vgg_normalised.t7')
parser.add_argument('--width', type=int, help='Webcam video width', default=None)
parser.add_argument('--height', type=int, help='Webcam video height', default=None)
parser.add_argument('--video-out', type=str, help="Save to output video file if not None", default=None)
parser.add_argument('--fps', type=int, help="Frames Per Second for output video file", default=10)
parser.add_argument('--scale', type=float, help="Scale the output image", default=1)
parser.add_argument('--keep-colors', action='store_true', help="Preserve the colors of the content image", default=False)
parser.add_argument('--passes', type=int, help="# of stylization passes per content image", default=1)
parser.add_argument('--device', type=str,
                        dest='device', help='Device to perform compute on',
                        default='/gpu:0')
parser.add_argument('--style-size', type=int, help="Resize style image to this size before cropping", default=512)
parser.add_argument('--crop-size', type=int, help="Crop a square from the style image", default=0)
parser.add_argument('--alpha', type=float, help="Alpha blend value for WCT features", default=1)
parser.add_argument('--concat', action='store_true', help="Concatenate style image and stylized output", default=False)
parser.add_argument('--noise', action='store_true', help="Synthesize textures from noise images", default=False)
parser.add_argument('-r', '--random', type=int, help='Load a random img every # iterations', default=0)
parser.add_argument('--adain', action='store_true', help="Use AdaIN instead of WCT", default=False)

## Style swap args
parser.add_argument('--swap5', action='store_true', help="Swap style on layer relu5_1", default=False)
parser.add_argument('--ss-alpha', type=float, help="Style swap alpha blend", default=0.6)
parser.add_argument('--ss-patch-size', type=int, help="Style swap patch size", default=3)
parser.add_argument('--ss-stride', type=int, help="Style swap stride", default=1)

args = parser.parse_args()


class StyleWindow(object):
    '''Helper class to handle style image settings'''

    def __init__(self, style_path, img_size=512, crop_size=512, scale=1, alpha=1, swap5=False, ss_alpha=1, passes=1):
        if os.path.isdir(style_path):
            self.style_imgs = get_files(style_path)
        else:
            self.style_imgs = [style_path]  # Single image instead of folder

        self.style_rgb = None

        self.img_size = img_size
        self.crop_size = crop_size
        self.scale = scale
        self.alpha = alpha
        self.ss_alpha = ss_alpha
        self.passes = passes

        cv2.namedWindow('Style Controls')
        if len(self.style_imgs) > 1:
            # Select style image by index
            cv2.createTrackbar('Index','Style Controls', 0, len(self.style_imgs)-1, self.set_idx)
        
        # Blend param for WCT/AdaIN transform
        cv2.createTrackbar('WCT/AdaIN alpha','Style Controls', int(self.alpha*100), 100, self.set_alpha)

        # Separate blend setting for style-swap
        cv2.createTrackbar('Style-swap alpha','Style Controls', int(self.ss_alpha*100), 100, self.set_ss_alpha)

        # Resize style to this size before cropping
        cv2.createTrackbar('Style size','Style Controls', self.img_size, 1280, self.set_size)

        # Size of square crop box for style
        cv2.createTrackbar('Style crop','Style Controls', self.crop_size, 1280, self.set_crop_size)

        # Scale the content before processing
        cv2.createTrackbar('Content scale','Style Controls', int(self.scale*100), 200, self.set_scale)

        # Num times to repeat the stylization pipeline
        cv2.createTrackbar('# of passes','Style Controls', self.passes, 5, self.set_passes)

        self.set_style(random=True, window='Style Controls')

    def set_style(self, idx=None, random=False, window='Style Controls'):
        if idx is not None:
            self.idx = idx
        if random:
            self.idx = np.random.randint(len(self.style_imgs))

        style_file = self.style_imgs[self.idx]
        print('Loading style image',style_file)
        if self.crop_size > 0:
            self.style_rgb = get_img_crop(style_file, resize=self.img_size, crop=self.crop_size)
        else:
            self.style_rgb = resize_to(get_img(style_file), self.img_size)
        self.show_style(window, self.style_rgb)

    def set_idx(self, idx):
        self.set_style(idx)

    def set_size(self, size):
        self.img_size = max(size, self.crop_size)      # Don't go below crop_size
        self.set_style()

    def set_crop_size(self, crop_size):
        self.crop_size = min(crop_size, self.img_size) # Don't go above img_size
        self.set_style()

    def set_scale(self, scale):
        self.scale = scale / 100
        
    def set_alpha(self, alpha):
        self.alpha = alpha / 100

    def show_style(self, window, style_rgb):
        cv2.imshow(window, cv2.cvtColor(cv2.resize(style_rgb, (args.style_size, args.style_size)), cv2.COLOR_RGB2BGR))

    # def set_interp(self, weight):
    #     self.interp_weight = weight / 100

    def set_ss_alpha(self, ss_alpha):
        self.ss_alpha = ss_alpha / 100

    def set_passes(self, passes):
        self.passes = passes


def main():
    # Load the WCT model
    wct_model = WCT(checkpoints=args.checkpoints, 
                    relu_targets=args.relu_targets,
                    vgg_path=args.vgg_path, 
                    device=args.device,
                    ss_patch_size=args.ss_patch_size, 
                    ss_stride=args.ss_stride)

    # Load a panel to control style settings
    style_window = StyleWindow(args.style_path, 
                               args.style_size, 
                               args.crop_size, 
                               args.scale, 
                               args.alpha, 
                               args.swap5, 
                               args.ss_alpha,
                               args.passes)

    # Start the webcam stream
    cap = WebcamVideoStream(args.video_source, args.width, args.height).start()

    _, frame = cap.read()

    # Grab a sample frame to calculate frame size
    frame_resize = cv2.resize(frame, None, fx=args.scale, fy=args.scale)
    img_shape = frame_resize.shape

    # Setup video out writer
    if args.video_out is not None:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        if args.concat:
            out_shape = (img_shape[1]+img_shape[0],img_shape[0]) # Make room for the style img
        else:
            out_shape = (img_shape[1],img_shape[0])
        print('Video Out Shape:', out_shape)
        video_writer = cv2.VideoWriter(args.video_out, fourcc, args.fps, out_shape)
    
    fps = FPS().start() # Track FPS processing speed

    # Toggles changed with kb shortcuts
    keep_colors = args.keep_colors
    swap_style = args.swap5
    use_adain = args.adain

    count = 0

    while(True):
        ret, frame = cap.read()

        if ret is True:       
            frame_resize = cv2.resize(frame, None, fx=style_window.scale, fy=style_window.scale)

            if args.noise:  # Generate textures from noise instead of images
                frame_resize = np.random.randint(0, 256, frame_resize.shape, np.uint8)

            count += 1
            print("Frame:",count,"Orig shape:",frame.shape,"New shape",frame_resize.shape)

            content_rgb = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR, we need RGB

            if args.random > 0 and count % args.random == 0:
                style_window.set_style(random=True)

            if keep_colors:
                style_rgb = preserve_colors_np(style_window.style_rgb, content_rgb)
            else:
                style_rgb = style_window.style_rgb

            # Run the frame through the style network
            stylized_rgb = wct_model.predict(content_rgb, style_rgb, style_window.alpha, swap_style, style_window.ss_alpha, use_adain)

            # Repeat stylization pipeline
            if style_window.passes > 1:
                for i in range(style_window.passes-1):
                    stylized_rgb = wct_model.predict(stylized_rgb, style_rgb, style_window.alpha, swap_style, style_window.ss_alpha, use_adain)

            # Stitch the style + stylized output together
            if args.concat:
                # Resize style img to same height as frame
                style_rgb_resized = cv2.resize(style_rgb, (stylized_rgb.shape[0], stylized_rgb.shape[0]))
                stylized_rgb = np.hstack([style_rgb_resized, stylized_rgb])
            
            stylized_bgr = cv2.cvtColor(stylized_rgb, cv2.COLOR_RGB2BGR)
                
            if args.video_out is not None:
                stylized_bgr = cv2.resize(stylized_bgr, out_shape) # Make sure frame matches video size
                video_writer.write(stylized_bgr)

            cv2.imshow('WCT Universal Style Transfer', stylized_bgr)

            fps.update()

            key = cv2.waitKey(10) 
            if key & 0xFF == ord('r'):   # Load new random style
                style_window.set_style(random=True)
            elif key & 0xFF == ord('c'): # Toggle color preservation
                keep_colors = not keep_colors
                print('Switching to keep_colors:',keep_colors)
            elif key & 0xFF == ord('s'): # Toggle style swap
                swap_style = not swap_style
                print('New value for flag swap_style:',swap_style)
            elif key & 0xFF == ord('a'): # Toggle AdaIN
                use_adain = not use_adain
                print('New value for flag use_adain:',use_adain)
            elif key & 0xFF == ord('w'): # Write stylized frame
                out_f = "{}.png".format(time.time())
                save_img(out_f, stylized_rgb)
                print('Saved image to:',out_f)
            elif key & 0xFF == ord('q'): # Quit gracefully
                break
        else:
            break

    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    cap.stop()
    
    if args.video_out is not None:
        video_writer.release()
    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
