from __future__ import division, print_function

import tensorflow as tf
import numpy as np
from vgg_normalised import vgg_from_t7
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, Lambda, UpSampling2D
from keras.initializers import VarianceScaling
from ops import pad_reflect, Conv2DReflect, torch_decay, mse, sse
import functools
from wct import wct_tf, wct_test


class AdaINModel(object):
    '''Adaptive Instance Normalization model from https://arxiv.org/abs/1703.06868'''

    def __init__(self, mode='train', vgg_weights=None, *args, **kwargs):
        # Build the graph
        self.build_model(vgg_weights)

        if mode == 'train':  # Train & summary ops only needed for training phase
            self.build_train(**kwargs)
            self.build_summary()

    def build_model(self, vgg_weights):
        batch_shape = (None, None, None, 3)

        self.compute_content =  tf.placeholder_with_default(tf.constant(True), shape=[])
        self.compute_style   =  tf.placeholder_with_default(tf.constant(False), shape=[])
        self.apply_wct       = tf.placeholder_with_default(tf.constant(False), shape=[])

        ### Load shared VGG model up to relu4_1
        with tf.name_scope('encoder'):
            self.vgg_model = vgg_from_t7(vgg_weights, target_layer='relu5_1')
        print(self.vgg_model.summary())

        ### Build encoder for reluX_1
        with tf.name_scope('content_layer_encoder'):
            self.content_imgs = tf.placeholder_with_default(tf.constant([[[[0.,0.,0.]]]]), shape=batch_shape, name='content_imgs')

            # Build content layer encoding model
            content_layer = self.vgg_model.get_layer('relu5_1').output
            self.content_encoder_model = Model(inputs=self.vgg_model.input, outputs=content_layer)

            # Setup content layer encodings for content images
            zeros = np.zeros((1,1,1,content_layer.get_shape()[-1]), dtype=np.float32)
            self.content_encoded = tf.cond(self.compute_content, lambda: self.content_encoder_model(self.content_imgs), lambda: tf.constant(zeros))

        with tf.name_scope('style_encoder'):
            self.style_img = tf.placeholder_with_default(self.content_imgs, shape=batch_shape, name='style_img')
            
            # self.style_encoded_pl = tf.placeholder_with_default(tf.zeros_like(content_layer.output), shape=batch_shape)
            # self.style_encoded_assign_pl = tf.placeholder_with_default(self.style_encoded, shape=batch_shape)
            # self.style_assign_op = tf.assign(self.style_encoded_pl, self.style_encoded_assign_pl)

            self.style_encoded = tf.cond(self.compute_style, lambda: self.content_encoder_model(self.style_img), lambda: self.content_encoded)
            
        ### Apply WCT if inference. During training pass through content_encoded unchanged.
        with tf.name_scope('wct'):
            self.alpha = tf.placeholder_with_default(1., shape=[], name='alpha')
            self.decoder_input = tf.cond(self.apply_wct, lambda: wct_tf(self.content_encoded, self.style_encoded, self.alpha), lambda: self.content_encoded)

        ### Build decoder
        with tf.name_scope('decoder'):
            n_channels = self.content_encoded.get_shape()[-1].value
            self.decoder_model = self.build_decoder(input_shape=(None, None, n_channels))

            self.decoder_input_wrapped = tf.placeholder_with_default(self.decoder_input, shape=[None,None,None,n_channels])
            
            # Reconstruct/decode from encoding
            self.decoded = self.decoder_model(Lambda(lambda x: x)(self.decoder_input_wrapped)) # Lambda converts TF tensor to Keras

        # Content layer encoding for stylized out
        self.decoded_encoded = self.content_encoder_model(self.decoded)

    def build_decoder(self, input_shape): 
        arch = [                                                            #  HxW  / InC->OutC
                Conv2DReflect(512, 3, padding='valid', activation='relu'),  # 32x32 / 512->256
                UpSampling2D(),
                Conv2DReflect(512, 3, padding='valid', activation='relu'),  # 32x32 / 512->256
                Conv2DReflect(512, 3, padding='valid', activation='relu'),  # 32x32 / 512->256
                Conv2DReflect(512, 3, padding='valid', activation='relu'),  # 32x32 / 512->256
                #Relu4_1
                Conv2DReflect(256, 3, padding='valid', activation='relu'),  # 32x32 / 512->256
                UpSampling2D(),                                             # 32x32 -> 64x64
                Conv2DReflect(256, 3, padding='valid', activation='relu'),  # 64x64 / 256->256
                Conv2DReflect(256, 3, padding='valid', activation='relu'),  # 64x64 / 256->256
                Conv2DReflect(256, 3, padding='valid', activation='relu'),  # 64x64 / 256->256
                Conv2DReflect(128, 3, padding='valid', activation='relu'),  # 64x64 / 256->128
                UpSampling2D(),                                             # 64x64 -> 128x128
                Conv2DReflect(128, 3, padding='valid', activation='relu'),  # 128x128 / 128->128
                Conv2DReflect(64, 3, padding='valid', activation='relu'),   # 128x128 / 128->64
                UpSampling2D(),                                             # 128x128 -> 256x256
                Conv2DReflect(64, 3, padding='valid', activation='relu'),   # 256x256 / 64->64
                Conv2DReflect(3, 3, padding='valid', activation=None)]      # 256x256 / 64->3
        
        code = Input(shape=input_shape, name='decoder_input')
        x = code

        with tf.variable_scope('decoder'):
            for layer in arch:
                x = layer(x)
            
        decoder = Model(code, x, name='decoder_model')
        print(decoder.summary())
        return decoder

    def build_train(self, 
                    batch_size=8,
                    content_weight=1, 
                    pixel_weight=1e-2, 
                    tv_weight=0,
                    learning_rate=1e-4, 
                    lr_decay=5e-5, 
                    use_gram=False):
        ### Extract style layer feature maps for input style & decoded stylized output
        # with tf.name_scope('style_layers'):
        #     # Build style model for blockX_conv1 tensors for X:[1,2,3,4]
        #     relu_layers = [ 'relu1_1',
        #                     'relu2_1',
        #                     'relu3_1',
        #                     'relu4_1' ]

        #     style_layers = [self.vgg_model.get_layer(l).output for l in relu_layers]
        #     self.style_layer_model = Model(inputs=self.vgg_model.input, outputs=style_layers)

        #     self.style_fmaps = self.style_layer_model(self.style_imgs)
        #     self.decoded_fmaps = self.style_layer_model(self.decoded)

        ### Losses
        with tf.name_scope('losses'):
            # Content loss between stylized encoding and AdaIN encoding
            self.content_loss = content_weight * mse(self.decoded_encoded, self.content_encoded)

            self.pixel_loss = pixel_weight * mse(self.decoded, self.content_imgs)

            # Total Variation loss
            if tv_weight > 0:
                self.tv_loss = tv_weight * tf.reduce_mean(tf.image.total_variation(self.decoded))
            else:
                self.tv_loss = tf.constant(0.)

            # Add it all together
            self.total_loss = self.content_loss + self.pixel_loss + self.tv_loss

        ### Training ops
        with tf.name_scope('train'):
            self.global_step = tf.Variable(0, name='global_step_train', trainable=False)
            # self.learning_rate = tf.train.exponential_decay(learning_rate, self.global_step, 100, 0.96, staircase=False)
            self.learning_rate = torch_decay(learning_rate, self.global_step, lr_decay)
            d_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.9)

            t_vars = tf.trainable_variables()
            self.d_vars = [var for var in t_vars if 'decoder' in var.name]  # Only train decoder vars, encoder is frozen

            self.train_op = d_optimizer.minimize(self.total_loss, var_list=self.d_vars, global_step=self.global_step)

    def build_summary(self):
        ### Loss & image summaries
        with tf.name_scope('summary'):
            content_loss_summary = tf.summary.scalar('content_loss', self.content_loss)
            pixel_loss_summary = tf.summary.scalar('pixel_loss', self.pixel_loss)
            tv_loss_summary = tf.summary.scalar('tv_loss', self.tv_loss)
            total_loss_summary = tf.summary.scalar('total_loss', self.total_loss)

            clip = lambda x: tf.clip_by_value(x, 0, 1)
            content_imgs_summary = tf.summary.image('content_imgs', self.content_imgs)
            # style_imgs_summary = tf.summary.image('style_imgs', clip(self.style_imgs))
            decoded_images_summary = tf.summary.image('decoded_images', clip(self.decoded))
            
            # # Visualize first three filters of encoding layers
            # sliced = lambda x: tf.slice(x, [0,0,0,0], [-1,-1,-1,3])
            # content_encoded_summary = tf.summary.image('content_encoded', sliced(self.content_encoded))
            # style_encoded_summary = tf.summary.image('style_encoded', sliced(self.style_encoded))
            # adain_encoded_summary = tf.summary.image('adain_encoded', sliced(self.adain_encoded))
            # decoded_encoded_summary = tf.summary.image('decoded_encoded', sliced(self.decoded_encoded))

            for var in self.d_vars:
                tf.summary.histogram(var.op.name, var)

            self.summary_op = tf.summary.merge_all()
