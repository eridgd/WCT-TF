from __future__ import division, print_function

import tensorflow as tf
from keras.layers import Conv2D, Lambda


### Layers ###

def pad_reflect(x, padding=1):
    return tf.pad(
      x, [[0, 0], [padding, padding], [padding, padding], [0, 0]],
      mode='REFLECT')

def Conv2DReflect(lambda_name, *args, **kwargs):
    return Lambda(lambda x: Conv2D(*args, **kwargs)(pad_reflect(x)), name=lambda_name)


### Misc ###

def torch_decay(learning_rate, global_step, decay_rate, name=None):
    '''Adapted from https://github.com/torch/optim/blob/master/adam.lua'''
    if global_step is None:
        raise ValueError("global_step is required for exponential_decay.")
    with tf.name_scope(name, "ExponentialDecay", [learning_rate, global_step, decay_rate]) as name:
        learning_rate = tf.convert_to_tensor(learning_rate, name="learning_rate")
        dtype = learning_rate.dtype
        global_step = tf.cast(global_step, dtype)
        decay_rate = tf.cast(decay_rate, dtype)

        # local clr = lr / (1 + state.t*lrd)
        return learning_rate / (1 + global_step*decay_rate)

# def gram_matrix(feature_maps):
#     """Computes the Gram matrix for a set of feature maps.
#        Borrowed from https://github.com/tensorflow/magenta/blob/9eb2e71074c09f55dba10cc493d26aef3168cdcb/magenta/models/image_stylization/learning.py
#     """
#     batch_size, height, width, channels = tf.unstack(tf.shape(feature_maps))
#     denominator = tf.to_float(height * width)
#     feature_maps = tf.reshape(
#       feature_maps, tf.stack([batch_size, height * width, channels]))
#     matrix = tf.matmul(feature_maps, feature_maps, adjoint_a=True)
#     return matrix / denominator
