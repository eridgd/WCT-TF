from __future__ import division, print_function

import tensorflow as tf
import numpy as np
from keras.layers import Conv2D, Lambda
from tensorflow.python.layers import utils
# from numba import jit


### Layers ###

def pad_reflect(x, padding=1):
    return tf.pad(
      x, [[0, 0], [padding, padding], [padding, padding], [0, 0]],
      mode='REFLECT')

def Conv2DReflect(lambda_name, *args, **kwargs):
    '''Wrap Keras Conv2D with reflect padding'''
    return Lambda(lambda x: Conv2D(*args, **kwargs)(pad_reflect(x)), name=lambda_name)


### Whiten-Color Transform ops ###

def wct_tf(content, style, alpha, eps=1e-8):
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
    fcfc = tf.matmul(fc, fc, transpose_b=True) / (tf.cast(Hc*Wc, tf.float32) - 1.) + tf.eye(Cc)*eps

    # Style covariance
    ms = tf.reduce_mean(style_flat, axis=1, keep_dims=True)
    fs = style_flat - ms
    fsfs = tf.matmul(fs, fs, transpose_b=True) / (tf.cast(Hs*Ws, tf.float32) - 1.) + tf.eye(Cs)*eps

    # tf.svd is slower on GPU, see https://github.com/tensorflow/tensorflow/issues/13603
    with tf.device('/cpu:0'):  
        Sc, Uc, _ = tf.svd(fcfc)
        Ss, Us, _ = tf.svd(fsfs)

    ## Uncomment to perform SVD for content/style with np in one call
    ## This is slower than CPU tf.svd but won't segfault for ill-conditioned matrices
    # @jit
    # def np_svd(content, style):
    #     '''tf.py_func helper to run SVD with NumPy for content/style cov tensors'''
    #     Uc, Sc, _ = np.linalg.svd(content)
    #     Us, Ss, _ = np.linalg.svd(style)
    #     return Uc, Sc, Us, Ss
    # Uc, Sc, Us, Ss = tf.py_func(np_svd, [fcfc, fsfs], [tf.float32, tf.float32, tf.float32, tf.float32])

    # Filter small singular values
    k_c = tf.reduce_sum(tf.cast(tf.greater(Sc, 1e-5), tf.int32))
    k_s = tf.reduce_sum(tf.cast(tf.greater(Ss, 1e-5), tf.int32))

    # Whiten content feature
    Dc = tf.diag(tf.pow(Sc[:k_c], -0.5))
    fc_hat = tf.matmul(tf.matmul(tf.matmul(Uc[:,:k_c], Dc), Uc[:,:k_c], transpose_b=True), fc)

    # Color content with style
    Ds = tf.diag(tf.pow(Ss[:k_s], 0.5))
    fcs_hat = tf.matmul(tf.matmul(tf.matmul(Us[:,:k_s], Ds), Us[:,:k_s], transpose_b=True), fc_hat)

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
    # 1xHxWxC -> CxHxW
    content_t = np.transpose(np.squeeze(content), (2, 0, 1))
    style_t = np.transpose(np.squeeze(style), (2, 0, 1))

    # CxHxW -> CxH*W
    content_flat = content_t.reshape(-1, content_t.shape[1]*content_t.shape[2])
    style_flat = style_t.reshape(-1, style_t.shape[1]*style_t.shape[2])

    mc = content_flat.mean(axis=1, keepdims=True)
    fc = content_flat - mc

    fcfc = np.dot(fc, fc.T) / (content_t.shape[1]*content_t.shape[2] - 1)
    
    Ec, wc, _ = np.linalg.svd(fcfc)

    k_c = (wc > 1e-5).sum()

    Dc = np.diag((wc[:k_c]+eps)**-0.5)

    fc_hat = Ec[:,:k_c].dot(Dc).dot(Ec[:,:k_c].T).dot(fc)

    ms = style_flat.mean(axis=1, keepdims=True)
    fs = style_flat - ms

    fsfs = np.dot(fs, fs.T) / (style_t.shape[1]*style_t.shape[2] - 1)

    Es, ws, _ = np.linalg.svd(fsfs)

    k_s = (ws > 1e-5).sum()
    
    Ds = np.sqrt(np.diag(ws[:k_s]+eps))

    fcs_hat = Es[:,:k_s].dot(Ds).dot(Es[:,:k_s].T).dot(fc_hat)

    fcs_hat = fcs_hat + ms

    blended = alpha*fcs_hat + (1 - alpha)*(fc)

    # CxH*W -> CxHxW
    blended = blended.reshape(content_t.shape)
    # CxHxW -> 1xHxWxC
    blended = np.expand_dims(np.transpose(blended, (1,2,0)), 0)
    
    return np.float32(blended)


### Style-Swap WCT ###

def wct_style_swap(content, style, alpha, patch_size=3, stride=1, eps=1e-8):
    '''Modified Whiten-Color Transform that performs style swap on whitened content/style encodings before coloring
       Assume that content/style encodings have shape 1xHxWxC
    '''
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
    fcfc = tf.matmul(fc, fc, transpose_b=True) / (tf.cast(Hc*Wc, tf.float32) - 1.) + tf.eye(Cc)*eps

    # Style covariance
    ms = tf.reduce_mean(style_flat, axis=1, keep_dims=True)
    fs = style_flat - ms
    fsfs = tf.matmul(fs, fs, transpose_b=True) / (tf.cast(Hs*Ws, tf.float32) - 1.) + tf.eye(Cs)*eps

    # tf.svd is slower on GPU, see https://github.com/tensorflow/tensorflow/issues/13603
    with tf.device('/cpu:0'):  
        Sc, Uc, _ = tf.svd(fcfc)
        Ss, Us, _ = tf.svd(fsfs)

    ## Uncomment to perform SVD for content/style with np in one call
    ## This is slower than CPU tf.svd but won't segfault for ill-conditioned matrices
    # @jit
    # def np_svd(content, style):
    #     '''tf.py_func helper to run SVD with NumPy for content/style cov tensors'''
    #     Uc, Sc, _ = np.linalg.svd(content)
    #     Us, Ss, _ = np.linalg.svd(style)
    #     return Uc, Sc, Us, Ss
    # Uc, Sc, Us, Ss = tf.py_func(np_svd, [fcfc, fsfs], [tf.float32, tf.float32, tf.float32, tf.float32])
    
    k_c = tf.reduce_sum(tf.cast(tf.greater(Sc, 1e-5), tf.int32))
    k_s = tf.reduce_sum(tf.cast(tf.greater(Ss, 1e-5), tf.int32))

    ### Whiten content
    Dc = tf.diag(tf.pow(Sc[:k_c], -0.5))

    fc_hat = tf.matmul(tf.matmul(tf.matmul(Uc[:,:k_c], Dc), Uc[:,:k_c], transpose_b=True), fc)

    # Reshape before passing to style swap, CxH*W -> 1xHxWxC
    whiten_content = tf.expand_dims(tf.transpose(tf.reshape(fc_hat, [Cc,Hc,Wc]), [1,2,0]), 0)

    ### Whiten style before swapping
    Ds = tf.diag(tf.pow(Ss[:k_s], -0.5))
    whiten_style = tf.matmul(tf.matmul(tf.matmul(Us[:,:k_s], Ds), Us[:,:k_s], transpose_b=True), fs)
    # Reshape before passing to style swap, CxH*W -> 1xHxWxC
    whiten_style = tf.expand_dims(tf.transpose(tf.reshape(whiten_style, [Cs,Hs,Ws]), [1,2,0]), 0)

    ### Style swap whitened encodings
    ss_feature = style_swap(whiten_content, whiten_style, patch_size, stride)
    # HxWxC -> CxH*W
    ss_feature = tf.transpose(tf.reshape(ss_feature, [Hc*Wc,Cc]), [1,0])

    ### Color style-swapped encoding with style 
    Ds_sq = tf.diag(tf.pow(Ss[:k_s], 0.5))
    fcs_hat = tf.matmul(tf.matmul(tf.matmul(Us[:,:k_s], Ds_sq), Us[:,:k_s], transpose_b=True), ss_feature)
    fcs_hat = fcs_hat + ms

    ### Blend style-swapped & colored encoding with original content encoding
    blended = alpha * fcs_hat + (1 - alpha) * (fc + mc)
    # CxH*W -> CxHxW
    blended = tf.reshape(blended, (Cc,Hc,Wc))
    # CxHxW -> 1xHxWxC
    blended = tf.expand_dims(tf.transpose(blended, (1,2,0)), 0)

    return blended

def style_swap(content, style, patch_size, stride):
    '''Efficiently swap content feature patches with nearest-neighbor style patches
       Original paper: https://arxiv.org/abs/1612.04337
       Adapted from: https://github.com/rtqichen/style-swap/blob/master/lib/NonparametricPatchAutoencoderFactory.lua
    '''
    nC = tf.shape(style)[-1]  # Num channels of input content feature and style-swapped output

    ### Extract patches from style image that will be used for conv/deconv layers
    style_patches = tf.extract_image_patches(style, [1,patch_size,patch_size,1], [1,stride,stride,1], [1,1,1,1], 'VALID')
    before_reshape = tf.shape(style_patches)  # NxRowsxColsxPatch_size*Patch_size*nC
    style_patches = tf.reshape(style_patches, [before_reshape[1]*before_reshape[2],patch_size,patch_size,nC])
    style_patches = tf.transpose(style_patches, [1,2,3,0])  # Patch_sizexPatch_sizexIn_CxOut_c

    # Normalize each style patch
    style_patches_norm = tf.nn.l2_normalize(style_patches, dim=3)

    # Compute cross-correlation/nearest neighbors of patches by using style patches as conv filters
    ss_enc = tf.nn.conv2d(content,
                          style_patches_norm,
                          [1,stride,stride,1],
                          'VALID')

    # For each spatial position find index of max along channel/patch dim  
    ss_argmax = tf.argmax(ss_enc, axis=3)
    encC = tf.shape(ss_enc)[-1]  # Num channels in intermediate conv output, same as # of patches
    
    # One-hot encode argmax with same size as ss_enc, with 1's in max channel idx for each spatial pos
    ss_oh = tf.one_hot(ss_argmax, encC, 1., 0., 3)

    # Calc size of transposed conv out
    deconv_out_H = utils.deconv_output_length(tf.shape(ss_oh)[1], patch_size, 'valid', stride)
    deconv_out_W = utils.deconv_output_length(tf.shape(ss_oh)[2], patch_size, 'valid', stride)
    deconv_out_shape = tf.stack([1,deconv_out_H,deconv_out_W,nC])

    # Deconv back to original content size with highest matching (unnormalized) style patch swapped in for each content patch
    ss_dec = tf.nn.conv2d_transpose(ss_oh,
                                    style_patches,
                                    deconv_out_shape,
                                    [1,stride,stride,1],
                                    'VALID')

    ### Interpolate to average overlapping patch locations
    ss_oh_sum = tf.reduce_sum(ss_oh, axis=3, keep_dims=True)

    filter_ones = tf.ones([patch_size,patch_size,1,1], dtype=tf.float32)
    
    deconv_out_shape = tf.stack([1,deconv_out_H,deconv_out_W,1])  # Same spatial size as ss_dec with 1 channel

    counting = tf.nn.conv2d_transpose(ss_oh_sum,
                                         filter_ones,
                                         deconv_out_shape,
                                         [1,stride,stride,1],
                                         'VALID')

    counting = tf.tile(counting, [1,1,1,nC])  # Repeat along channel dim to make same size as ss_dec

    interpolated_dec = tf.divide(ss_dec, counting)

    return interpolated_dec

### Adaptive Instance Normalization ###

def adain(content_features, style_features, alpha, epsilon=1e-5):
    '''
    Borrowed from https://github.com/jonrei/tf-AdaIN
    Normalizes the `content_features` with scaling and offset from `style_features`.
    See "5. Adaptive Instance Normalization" in https://arxiv.org/abs/1703.06868 for details.
    '''
    style_mean, style_variance = tf.nn.moments(style_features, [1,2], keep_dims=True)
    content_mean, content_variance = tf.nn.moments(content_features, [1,2], keep_dims=True)
    normalized_content_features = tf.nn.batch_normalization(content_features, content_mean,
                                                            content_variance, style_mean, 
                                                            tf.sqrt(style_variance), epsilon)
    normalized_content_features = alpha * normalized_content_features + (1 - alpha) * content_features
    return normalized_content_features

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
