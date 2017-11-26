from __future__ import division, print_function

import tensorflow as tf
import numpy as np
from keras.layers import Conv2D, Lambda


### Layers ###

def pad_reflect(x, padding=1):
    return tf.pad(
      x, [[0, 0], [padding, padding], [padding, padding], [0, 0]],
      mode='REFLECT')

def Conv2DReflect(lambda_name, *args, **kwargs):
    '''Wrap Keras Conv2D with reflect padding'''
    return Lambda(lambda x: Conv2D(*args, **kwargs)(pad_reflect(x)), name=lambda_name)


### Whiten-Color Transform ops ###

def np_svd(content, style):
    '''tf.py_func helper to run SVD with NumPy for content/style cov tensors'''
    Uc, Sc, _ = np.linalg.svd(content)
    Us, Ss, _ = np.linalg.svd(style)
    return Uc, Sc, Us, Ss

def wct_tf(content, style, alpha, eps=1e-5):
    '''TensorFlow version of Whiten-Color Transform
       Assume that content/style encodings have shape 1xHxWxC

       See p.4 of the Universal Style Transfer paper for corresponding equations:
       https://arxiv.org/pdf/1705.08086.pdf
    '''
    # Remove batch dim and reorder to CxHxW
    content_t = tf.transpose(tf.squeeze(content), (2, 0, 1))
    style_t = tf.transpose(tf.squeeze(style), (2, 0, 1))

    Cc, Hc, Wc = tf.unstack(tf.shape(content_t))
    Cs, Hs, Ws = tf.unstack(tf.shape(style_t))

    # CxHxW -> CxH*W
    content_flat = tf.reshape(content_t, (Cc, Hc*Wc))
    style_flat = tf.reshape(style_t, (Cs, Hs*Ws))

    # Content covariance
    mc = tf.reduce_mean(content_flat, axis=1, keep_dims=True)
    fc = content_flat - mc
    fcfc = tf.matmul(fc, fc, transpose_b=True) / (tf.cast(Hc*Wc, tf.float32) - 1.)

    # Style covariance
    ms = tf.reduce_mean(style_flat, axis=1, keep_dims=True)
    fs = style_flat - ms
    fsfs = tf.matmul(fs, fs, transpose_b=True) / (tf.cast(Hs*Ws, tf.float32) - 1.)

    # Perform SVD for content/style with np in one call to limit memory copy overhead
    Uc, Sc, Us, Ss = tf.py_func(np_svd, [fcfc, fsfs], [tf.float32, tf.float32, tf.float32, tf.float32])

    ## Uncomment to calculate SVD using TF. On CPU this is fast but unstable - https://github.com/tensorflow/tensorflow/issues/9234
    ## If using TF r1.4 this can be changed to /gpu:0, it's stable but very slow - https://github.com/tensorflow/tensorflow/issues/13603
    # with tf.device('/cpu:0'):  
    #     Sc, Uc, _ = tf.svd(fcfc)
    #     Ss, Us, _ = tf.svd(fsfs)

    # Whiten content feature
    Dc = tf.diag(tf.pow(Sc + eps, -0.5))
    fc_hat = tf.matmul(tf.matmul(tf.matmul(Uc, Dc), Uc, transpose_b=True), fc)

    # Color content with style
    Ds = tf.diag(tf.pow(Ss + eps, 0.5))
    fcs_hat = tf.matmul(tf.matmul(tf.matmul(Us, Ds), Us, transpose_b=True), fc_hat)

    # Re-center with mean of style
    fcs_hat = fcs_hat + ms

    # Blend whiten-colored feature with original content feature
    blended = alpha * fcs_hat + (1 - alpha) * (fc + mc)

    # CxH*W -> CxHxW
    blended = tf.reshape(blended, (Cc,Hc,Wc))
    # CxHxW -> 1xHxWxC
    blended = tf.expand_dims(tf.transpose(blended, (1,2,0)), 0)

    return blended

def wct_np(content, style, alpha=0.6, eps=1e-5):
    '''Perform Whiten-Color Transform on feature maps using numpy
       See p.4 of the Universal Style Transfer paper for equations:
       https://arxiv.org/pdf/1705.08086.pdf
    '''    
    # HxWxC -> CxHxW
    content_t = np.transpose(content, (2, 0, 1))
    style_t = np.transpose(style, (2, 0, 1))

    # CxHxW -> CxH*W
    content_flat = content_t.reshape(-1, content_t.shape[1]*content_t.shape[2])
    style_flat = style_t.reshape(-1, style_t.shape[1]*style_t.shape[2])

    mc = content_flat.mean(axis=1, keepdims=True)
    fc = content_flat - mc

    fcfc = np.dot(fc, fc.T) / (content_t.shape[1]*content_t.shape[2] - 1)
    
    Ec, wc, _ = np.linalg.svd(fcfc)

    Dc_sq_inv = np.linalg.inv(np.sqrt(np.diag(wc+eps)))

    fc_hat = Ec.dot(Dc_sq_inv).dot(Ec.T).dot(fc)

    ms = style_flat.mean(axis=1, keepdims=True)
    fs = style_flat - ms

    fsfs = np.dot(fs, fs.T) / (style_t.shape[1]*style_t.shape[2] - 1)

    Es, ws, _ = np.linalg.svd(fsfs)
    
    Ds_sq = np.sqrt(np.diag(ws+eps))

    fcs_hat = Es.dot(Ds_sq).dot(Es.T).dot(fc_hat)

    fcs_hat = fcs_hat + ms

    blended = alpha*fcs_hat + (1 - alpha)*(fc)

    # CxH*W -> CxHxW
    blended = blended.reshape(content_t.shape)

    # CxHxW -> HxWxC
    blended = np.transpose(blended, (1,2,0))  
    
    return blended


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
