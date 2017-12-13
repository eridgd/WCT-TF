import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, UpSampling2D, Activation, Lambda, MaxPooling2D
from ops import pad_reflect
import torchfile


def vgg_from_t7(t7_file, target_layer=None):
    '''Extract VGG layers from a Torch .t7 model into a Keras model
       e.g. vgg = vgg_from_t7('vgg_normalised.t7', target_layer='relu4_1')
       Adapted from https://github.com/jonrei/tf-AdaIN/blob/master/AdaIN.py
       Converted caffe->t7 from https://github.com/xunhuang1995/AdaIN-style
    '''
    t7 = torchfile.load(t7_file, force_8bytes_long=True)
    
    inp = Input(shape=(None, None, 3), name='vgg_input')

    x = inp
    
    for idx,module in enumerate(t7.modules):
        name = module.name.decode() if module.name is not None else None
        
        if idx == 0:
            name = 'preprocess'  # VGG 1st layer preprocesses with a 1x1 conv to multiply by 255 and subtract BGR mean as bias

        if module._typename == b'nn.SpatialReflectionPadding':
            x = Lambda(pad_reflect)(x)            
        elif module._typename == b'nn.SpatialConvolution':
            filters = module.nOutputPlane
            kernel_size = module.kH
            weight = module.weight.transpose([2,3,1,0])
            bias = module.bias
            x = Conv2D(filters, kernel_size, padding='valid', activation=None, name=name,
                        kernel_initializer=lambda shape: K.constant(weight, shape=shape),
                        bias_initializer=lambda shape: K.constant(bias, shape=shape),
                        trainable=False)(x)
        elif module._typename == b'nn.ReLU':
            x = Activation('relu', name=name)(x)
        elif module._typename == b'nn.SpatialMaxPooling':
            x = MaxPooling2D(padding='same', name=name)(x)
        # elif module._typename == b'nn.SpatialUpSamplingNearest': # Not needed for VGG
        #     x = Upsampling2D(name=name)(x)
        else:
            raise NotImplementedError(module._typename)

        if name == target_layer:
            # print("Reached target layer", target_layer)
            break
    
    # Hook it up
    model = Model(inputs=inp, outputs=x)

    return model
