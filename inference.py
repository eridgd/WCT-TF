from __future__ import division, print_function

import os
import numpy as np
import time
from model import AdaINModel
import tensorflow as tf
from wct import wct_np
from coral import coral_numpy


class AdaINference(object):
    '''Styilze images with trained AdaIN model'''

    def __init__(self, checkpoints, vgg_weights, device='/gpu:0'): 
        '''
            Args:
                checkpoint_dir: Path to trained model checkpoint
                device: String for device ID to load model onto
        '''       
        graph = tf.get_default_graph()

        relu_targets = ['relu5_1','relu4_1']

        with graph.device(device):
            self.model = AdaINModel(mode='test', relu_targets=['relu5_1','relu4_1'], vgg_weights=vgg_weights)
            
            self.content_input = self.model.content_input
            # self.encoded = self.model.
            self.decoded_output = self.model.decoded_output
            self.style_encoded = None

            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            self.sess = sess

            self.sess.run(tf.global_variables_initializer())

            ckpts = ['/home/rachael/Downloads/tflogs/wct/multi_relu5_1_f1e-2/','/home/rachael/Downloads/tflogs/wct/multi_relu4_1_f1e-2/']

            for relu_target, checkpoint_dir in zip(relu_targets, ckpts):
                decoder_prefix = 'decoder_{}'.format(relu_target)
                relu_vars = [v for v in tf.trainable_variables() if decoder_prefix in v.name]

                saver = tf.train.Saver(var_list=relu_vars)
                
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    print("Restoring vars for {} from checkpoint {}".format(relu_target, ckpt.model_checkpoint_path))
                    saver.restore(self.sess, ckpt.model_checkpoint_path)
                else:
                    raise Exception("No checkpoint found...")

    @staticmethod
    def preprocess(image):
        if len(image.shape) == 3:  # Add batch dimension
            image = np.expand_dims(image, 0)
        return image / 255.        # Range [0,1]

    @staticmethod
    def postprocess(image):
        return np.uint8(np.clip(image, 0, 1) * 255)

    def predict(self, content, style, alpha=1):
        '''Stylize a single content/style pair
           Assumes that images are RGB [0,255]
        '''
        content = self.preprocess(content)
        style   = self.preprocess(style)

        # if self.style_encoded is None:
        #     self.style_encoded = self.sess.run(self.model.style_encoded, feed_dict={self.model.style_img: style,
        #                                                                       self.model.compute_style: True})
        #     print("Computed style encoded")

        # content_encoded = self.sess.run(self.encoded, feed_dict={self.content_input: content})
        s = time.time()
        stylized = self.sess.run(self.decoded_output, feed_dict={
                                                          self.content_input: content,
                                                          self.model.style_input: style,
                                                          self.model.compute_content: True,
                                                          # self.model.compute_style: True,
                                                          self.model.apply_wct: True,
                                                          self.model.alpha: alpha})
        print(time.time() - s)
        # style_encoded   = self.sess.run(self.encoded, feed_dict={self.content_input: style})

        # encoded_wct = wct(content_encoded.squeeze(), style_encoded.squeeze(), alpha)

        # stylized = self.sess.run(self.decoded, feed_dict={self.encoded_pl: np.expand_dims(encoded_wct, 0)})

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


    # def predict_batch(self, content_batch, style, alpha=1):
    #     '''Stylize a batch of content imgs with a single style
    #        Assumes that images are RGB [0,255]
    #     '''
    #     content_batch = self.preprocess(content_batch)
    #     style_batch = np.stack([style]*len(content_batch)) 
    #     style_batch = self.preprocess(style_batch)

    #     stylized = self.sess.run(self.stylized, feed_dict={self.content_input: content_batch,
    #                                                        self.style_imgs:   style_batch,
    #                                                        self.alpha_tensor: alpha})

    #     return self.postprocess(stylized)

    # def predict_interpolate(self, content, styles, style_weights, alpha=1):
    #     '''Stylize a weighted sum of multiple style encodings for a single content'''
    #     content_stacked = np.stack([content]*len(styles))  # Repeat content for each style
    #     style_stacked = np.stack(styles)
    #     content_stacked = self.preprocess(content_stacked)
    #     style_stacked = self.preprocess(style_stacked)

    #     encoded = self.sess.run(self.model.adain_encoded, feed_dict={self.content_input: content_stacked,
    #                                                                  self.style_imgs:   style_stacked,
    #                                                                  self.alpha_tensor: alpha})

    #     # Weight & combine AdaIN transformed encodings
    #     style_weights = np.array(style_weights).reshape((-1, 1, 1, 1))
    #     encoded_weighted = encoded * style_weights
    #     encoded_interpolated = np.sum(encoded_weighted, axis=0, keepdims=True)

    #     stylized = self.sess.run(self.stylized, feed_dict={self.model.adain_encoded_pl: encoded_interpolated})

    #     return self.postprocess(stylized[0])
