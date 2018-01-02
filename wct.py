from __future__ import division, print_function

import os
import numpy as np
import time
from model import WCTModel
import tensorflow as tf
from ops import wct_np
from coral import coral_numpy
import scipy.misc
from utils import swap_filter_fit, center_crop_to


class WCT(object):
    '''Styilze images with trained WCT model'''

    def __init__(self, checkpoints, relu_targets, vgg_path, device='/gpu:0',
                 ss_patch_size=3, ss_stride=1): 
        '''
            Args:
                checkpoints: List of trained decoder model checkpoint dirs
                relu_targets: List of relu target layers corresponding to decoder checkpoints
                vgg_path: Normalised VGG19 .t7 path
                device: String for device ID to load model onto
        '''       
        self.ss_patch_size = ss_patch_size
        self.ss_stride = ss_stride

        graph = tf.get_default_graph()

        with graph.device(device):
            # Build the graph
            self.model = WCTModel(mode='test', relu_targets=relu_targets, vgg_path=vgg_path,
                                  ss_patch_size=self.ss_patch_size, ss_stride=self.ss_stride)
            
            self.content_input = self.model.content_input
            self.decoded_output = self.model.decoded_output
            # self.style_encoded = None

            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)

            self.sess.run(tf.global_variables_initializer())

            # Load decoder vars one-by-one into the graph
            for relu_target, checkpoint_dir in zip(relu_targets, checkpoints):
                decoder_prefix = 'decoder_{}'.format(relu_target)
                relu_vars = [v for v in tf.trainable_variables() if decoder_prefix in v.name]

                saver = tf.train.Saver(var_list=relu_vars)
                
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    print('Restoring vars for {} from checkpoint {}'.format(relu_target, ckpt.model_checkpoint_path))
                    saver.restore(self.sess, ckpt.model_checkpoint_path)
                else:
                    raise Exception('No checkpoint found for target {} in dir {}'.format(relu_target, checkpoint_dir))

    @staticmethod
    def preprocess(image):
        if len(image.shape) == 3:  # Add batch dimension
            image = np.expand_dims(image, 0)
        return image / 255.        # Range [0,1]

    @staticmethod
    def postprocess(image):
        return np.uint8(np.clip(image, 0, 1) * 255)

    def predict(self, content, style, alpha=1, swap5=False, ss_alpha=1, adain=False):
        '''Stylize a single content/style pair.

           Args:
               content: Array for content image in [0,255]
               style: Array for style image in [0,255]
               alpha: Float blending value for WCT op in [0,1]
               swap5: If True perform style swap at layer relu5_1 instead of WCT
               ss_alpha: [0,1] Float blending value for style-swapped feature & content feature
               adain: Boolean indicating whether to use AdaIN transform instead of WCT
           Returns:
               Stylized image with pixels in [0,255]
        '''
        # If doing style swap and stride > 1 the content might need to be resized for the filter to fit
        if swap5 is True and self.ss_stride != 1:
            old_H, old_W = content.shape[:2]

            should_refit, H, W = swap_filter_fit(old_H, old_W, self.ss_patch_size, self.ss_stride)

            if should_refit:
                content = center_crop_to(content, H, W)

        # Make sure shape is correct and pixels are in [0,1]
        content = self.preprocess(content)
        style   = self.preprocess(style)

        s = time.time()
        stylized = self.sess.run(self.decoded_output, feed_dict={
                                                          self.content_input: content,
                                                          self.model.style_input: style,
                                                          self.model.alpha: alpha,
                                                          self.model.swap5: swap5,
                                                          self.model.ss_alpha: ss_alpha,
                                                          self.model.use_adain: adain})
        print("Stylized in:",time.time() - s)

        return self.postprocess(stylized[0])
