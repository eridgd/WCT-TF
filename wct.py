from __future__ import division, print_function

import os
import numpy as np
import time
from model import WCTModel
import tensorflow as tf
from ops import wct_np
from coral import coral_numpy


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
        graph = tf.get_default_graph()

        with graph.device(device):
            # Build the graph
            self.model = WCTModel(mode='test', relu_targets=relu_targets, vgg_path=vgg_path,
                                  ss_patch_size=ss_patch_size, ss_stride=ss_stride)
            
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

    def predict(self, content, style, alpha=1,
                swap5=False, ss_alpha=1., ss_patch_size=3, ss_stride=1):
        '''Stylize a single content/style pair
           Assumes that images are RGB [0,255]
        '''
        content = self.preprocess(content)
        style   = self.preprocess(style)

        s = time.time()
        stylized = self.sess.run(self.decoded_output, feed_dict={
                                                          self.content_input: content,
                                                          self.model.style_input: style,
                                                          # self.model.apply_wct: True,
                                                          self.model.alpha: alpha,
                                                          self.model.swap5: swap5,
                                                          self.model.ss_alpha: ss_alpha})
        print(time.time() - s)

        return self.postprocess(stylized[0])

    # def predict_np(self, content, style, alpha=1):
    #     '''Stylize a single content/style pair with numpy WCT
    #        Assumes that images are RGB [0,255]
    #     '''
    #     content = self.preprocess(content)
    #     style   = self.preprocess(style)

    #     # if self.style_encoded is None:
    #     style_encoded = self.sess.run(self.model.style_encoded, feed_dict={self.model.style_input: style,
    #                                                                        self.model.compute_style: True,
    #                                                                        self.model.compute_content: False})

        
    #     s = time.time()
    #     # print("style")
    #     # print(style_encoded)
    #     content_encoded = self.sess.run(self.encoded, feed_dict={self.model.content_input: content,
    #                                                              self.model.compute_content: True})
    #     # print("content")
    #     # print(content_encoded)
    #     encoded_wct = wct_np(content_encoded.squeeze(), style_encoded.squeeze(), alpha)
    #     # print("wct")
    #     # print(encoded_wct)

    #     stylized = self.sess.run(self.decoded, feed_dict={self.model.decoder_input: np.expand_dims(encoded_wct, 0)})
    #     # stylized = self.sess.run(self.decoded, feed_dict={
    #     #                                                   self.content_input: content,
    #     #                                                   self.model.style_img: style,
    #     #                                                   self.model.compute_content: True,
    #     #                                                   self.model.compute_style: True,
    #     #                                                   self.model.apply_wct: True,
    #     #                                                   self.model.alpha: alpha})
    #     print(time.time() - s)
        

    #     return self.postprocess(stylized[0])

    # def predict_interpolate(self, content, styles, style_weights, alpha=1):
    #     '''Stylize a weighted sum of multiple style encodings for a single content'''
    #     content_stacked = np.stack([content]*len(styles))  # Repeat content for each style
    #     style_stacked = np.stack(styles)
    #     content_stacked = self.preprocess(content_stacked)
    #     style_stacked = self.preprocess(style_stacked)

    #     encoded = self.sess.run(self.model.wct_encoded, feed_dict={self.content_input: content_stacked,
    #                                                                  self.style_imgs:   style_stacked,
    #                                                                  self.alpha_tensor: alpha})

    #     # Weight & combine WCT transformed encodings
    #     style_weights = np.array(style_weights).reshape((-1, 1, 1, 1))
    #     encoded_weighted = encoded * style_weights
    #     encoded_interpolated = np.sum(encoded_weighted, axis=0, keepdims=True)

    #     stylized = self.sess.run(self.stylized, feed_dict={self.model.wct_encoded_pl: encoded_interpolated})

    #     return self.postprocess(stylized[0])
