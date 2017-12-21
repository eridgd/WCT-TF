from __future__ import print_function, division

import scipy.misc, numpy as np, os, sys
import random
from coral import coral_numpy # , coral_pytorch
# from color_transfer import color_transfer


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

def center_crop_to(img, H_target, W_target):
    '''Center crop a rectangle of given dimensions and resize if necessary'''
    height, width = img.shape[0], img.shape[1]

    if height < H_target or width < W_target:
        H_rat, W_rat = H_target / height, W_target / width
        rat = max(H_rat, W_rat)

        img = scipy.misc.imresize(img, rat, interp='bilinear')
        height, width = img.shape[0], img.shape[1]

    h_off = (height - H_target) // 2
    w_off = (width - W_target) // 2
    return img[h_off:h_off+H_target,w_off:w_off+W_target]

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
    
    return scipy.misc.imresize(img, resize_shape, interp='bilinear')

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

def preserve_colors_np(style_rgb, content_rgb):
    coraled = coral_numpy(style_rgb/255., content_rgb/255.)
    coraled = np.uint8(np.clip(coraled, 0, 1) * 255.)
    return coraled

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

# def preserve_colors_pytorch(style_rgb, content_rgb):
#     coraled = coral_pytorch(style_rgb/255., content_rgb/255.)
#     coraled = np.uint8(np.clip(coraled, 0, 1) * 255.)
#     return coraled

# def preserve_colors_color_transfer(style_rgb, content_rgb):
#     style_bgr = cv2.cvtColor(style_rgb, cv2.COLOR_RGB2BGR)
#     content_bgr = cv2.cvtColor(content_rgb, cv2.COLOR_RGB2BGR)
#     transferred = color_transfer(content_bgr, style_bgr)
#     return cv2.cvtColor(transferred, cv2.COLOR_BGR2RGB)

def swap_filter_fit(H, W, patch_size, stride, n_pools=4):
    '''Style swap may not output same size encoding if filter size > 1, calculate a new size to avoid this'''
    # Calculate size of encodings after max pooling n_pools times
    pool_out_size = lambda x: (x + 2 - 1) // 2    
    H_pool_out, W_pool_out = H, W
    for _ in range(n_pools):
        H_pool_out, W_pool_out = pool_out_size(H_pool_out), pool_out_size(W_pool_out)
    
    # Size of encoding after applying conv to determine nearest neighbor patches
    H_conv_out = (H_pool_out - patch_size) // stride + 1
    W_conv_out = (W_pool_out - patch_size) // stride + 1

    # Size after transposed conv
    H_deconv_out = (H_conv_out - 1) * stride + patch_size
    W_deconv_out = (W_conv_out - 1) * stride + patch_size

    # Stylized output size after decoding
    H_out = H_deconv_out * 2**n_pools
    W_out = W_deconv_out * 2**n_pools

    # Image will need to be resized/cropped if pooled encoding does not match style-swap encoding in either dim
    should_refit = (H_pool_out != H_deconv_out) or (W_pool_out != W_deconv_out)

    return should_refit, H_out, W_out
