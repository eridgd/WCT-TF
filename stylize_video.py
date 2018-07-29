# Adapted from https://github.com/lengstrom/fast-style-transfer/blob/master/evaluate.py
from __future__ import print_function,division

import argparse
import sys
import os, random, subprocess, shutil, time
import numpy as np
import json
import scipy
from utils import preserve_colors_np
from utils import get_files, get_img, get_img_crop, save_img, resize_to, center_crop
from wct import WCT

TMP_DIR = '_____fns_frames_%s/' % random.randint(0,99999)

parser = argparse.ArgumentParser()

parser.add_argument('--checkpoints', nargs='+', type=str, help='List of checkpoint directories', required=True)
parser.add_argument('--relu-targets', nargs='+', type=str, help='List of reluX_1 layers, corresponding to --checkpoints', required=True)
parser.add_argument('--vgg-path', type=str, help='Path to vgg_normalised.t7', default='models/vgg_normalised.t7')
parser.add_argument('--in-path', type=str, help='Path to video files', required=True)
parser.add_argument('--out-path', type=str, help='Output video file path', required=True)
parser.add_argument('--style-path', type=str, help='Path to style image', required=True)
parser.add_argument('--tmp-dir', type=str, dest='tmp_dir', help='tmp dir for processing', default=TMP_DIR)
parser.add_argument('--keep-tmp', action='store_true', help='Don\'t remove stylized image tmp dir after', default=False)
parser.add_argument('--keep-colors', action='store_true', help="Preserve the colors of the style image", default=False)
parser.add_argument('--style-size', type=int, help="Resize style image to this size before cropping, default 512", default=0)
parser.add_argument('--crop-size', type=int, help="Crop square size, default 256", default=0)
parser.add_argument('--content-size', type=int, help="Resize short side of content image to this", default=0)
parser.add_argument('--passes', type=int, help="# of stylization passes per content image", default=1)
parser.add_argument('--device', type=str, help='Device to perform compute on, e.g. /gpu:0', default='/gpu:0')
parser.add_argument('--alpha', type=float, help="Alpha blend value", default=1)
parser.add_argument('--concat', action='store_true', help="Concatenate style image and stylized output", default=False)

## Style swap args
parser.add_argument('--swap5', action='store_true', help="Swap style on layer relu5_1", default=False)
parser.add_argument('--ss-alpha', type=float, help="Style swap alpha blend", default=0.6)
parser.add_argument('--ss-patch-size', type=int, help="Style swap patch size", default=3)
parser.add_argument('--ss-stride', type=int, help="Style swap stride", default=1)

args = parser.parse_args()


def main():
    # Load the WCT model
    wct_model = WCT(checkpoints=args.checkpoints, 
                                relu_targets=args.relu_targets,
                                vgg_path=args.vgg_path, 
                                device=args.device,
                                ss_patch_size=args.ss_patch_size, 
                                ss_stride=args.ss_stride)

    # Create needed dirs
    in_dir = os.path.join(args.tmp_dir, 'input')
    out_dir = os.path.join(args.tmp_dir, 'sytlized')
    if not os.path.exists(in_dir):
        os.makedirs(in_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if os.path.isdir(args.in_path):
        in_path = get_files(args.in_path)
    else: # Single image file
        in_path = [args.in_path]

    if os.path.isdir(args.style_path):
        style_files = get_files(args.style_path)
    else: # Single image file
        style_files = [args.style_path]

    print(style_files)
    import time
    # time.sleep(999)

    in_args = [
        'ffmpeg',
        '-i', args.in_path,
        '%s/frame_%%d.png' % in_dir
    ]

    subprocess.call(" ".join(in_args), shell=True)
    base_names = os.listdir(in_dir)
    in_files = [os.path.join(in_dir, x) for x in base_names]
    out_files = [os.path.join(out_dir, x) for x in base_names]

    


    s = time.time()
    for content_fullpath in in_path:
        content_prefix, content_ext = os.path.splitext(content_fullpath)
        content_prefix = os.path.basename(content_prefix)


        try:

            for style_fullpath in style_files:
                style_img = get_img(style_fullpath)
                if args.style_size > 0:
                    style_img = resize_to(style_img, args.style_size)
                if args.crop_size > 0:
                    style_img = center_crop(style_img, args.crop_size)

                style_prefix, _ = os.path.splitext(style_fullpath)
                style_prefix = os.path.basename(style_prefix)

                # print("ARRAY:  ", style_img)
                out_v = os.path.join(args.out_path, '{}_{}{}'.format(content_prefix, style_prefix, content_ext))
                print("OUT:",out_v)
                if os.path.isfile(out_v):
                    print("SKIP" , out_v)
                    continue
                
                for in_f, out_f in zip(in_files, out_files):
                    print('{} -> {}'.format(in_f, out_f))
                    content_img = get_img(in_f)

                    if args.keep_colors:
                        style_rgb = preserve_colors_np(style_img, content_img)
                    else:
                        style_rgb = style_img

                    stylized = wct_model.predict(content_img, style_rgb, args.alpha, args.swap5, args.ss_alpha)

                    if args.passes > 1:
                        for _ in range(args.passes-1):
                            stylized = wct_model.predict(stylized, style_rgb, args.alpha)

                    # Stitch the style + stylized output together, but only if there's one style image
                    if args.concat:
                        # Resize style img to same height as frame
                        style_img_resized = scipy.misc.imresize(style_rgb, (stylized.shape[0], stylized.shape[0]))
                        stylized = np.hstack([style_img_resized, stylized])

                    save_img(out_f, stylized)

                fr = 30
                out_args = [
                    'ffmpeg',
                    '-i', '%s/frame_%%d.png' % out_dir,
                    '-f', 'mp4',
                    '-q:v', '0',
                    '-vcodec', 'mpeg4',
                    '-r', str(fr),
                    '"' + out_v + '"'
                ]
                print(out_args)

                subprocess.call(" ".join(out_args), shell=True)
                print('Video at: %s' % out_v)

                if args.keep_tmp is True or len(style_files) > 1:
                    continue
                else:
                    shutil.rmtree(args.tmp_dir)
                print('Processed in:',(time.time() - s))

            print('Processed in:',(time.time() - s))
 
        except Exception as e:
            print("EXCEPTION: ",e)
            # main()

if __name__ == '__main__':
    main()
