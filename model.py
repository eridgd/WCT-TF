from __future__ import division, print_function

import tensorflow as tf
import numpy as np
from vgg_normalised import vgg_from_t7
from keras import backend as K
from keras.models import Model
from keras.layers import Input, UpSampling2D, Lambda
from ops import pad_reflect, Conv2DReflect, torch_decay, wct_tf, wct_style_swap, adain
from collections import namedtuple


### Helpers ###

mse = tf.losses.mean_squared_error

clip = lambda x: tf.clip_by_value(x, 0, 1)

EncoderDecoder = namedtuple('EncoderDecoder', 
                            'content_input content_encoder_model content_encoded \
                             style_encoded \
                             decoder_input, decoder_model decoded decoded_encoded \
                             pixel_loss feature_loss tv_loss total_loss \
                             train_op learning_rate global_step \
                             summary_op')


### WCT Model Graph ###

class WCTModel(object):
    '''Model graph for Universal Style Transfer via Feature Transforms from https://arxiv.org/abs/1705.08086'''

    def __init__(self, mode='train', relu_targets=['relu5_1','relu4_1','relu3_1','relu2_1','relu1_1'], vgg_path=None,  
                 *args, **kwargs):
        '''
            Args:
                mode: 'train' or 'test'. If 'train' then training & summary ops will be added to the graph
                relu_targets: List of relu target layers corresponding to decoder checkpoints
                vgg_path: Normalised VGG19 .t7 path
        '''
        self.mode = mode

        self.style_input = tf.placeholder_with_default(tf.constant([[[[0.,0.,0.]]]]), shape=(None, None, None, 3), name='style_img')

        self.alpha = tf.placeholder_with_default(1., shape=[], name='alpha')
        
        # Style swap settings
        self.swap5 = tf.placeholder_with_default(tf.constant(False), shape=[])
        self.ss_alpha = tf.placeholder_with_default(.7, shape=[], name='ss_alpha')

        # Flag to use AdaIN instead of WCT
        self.use_adain = tf.placeholder_with_default(tf.constant(False), shape=[])
        
        self.encoder_decoders = []
        
        ### Build the graph ###
        
        # Load shared VGG model up to deepest target layer
        with tf.name_scope('vgg_encoder'):
            deepest_target = sorted(relu_targets)[-1]
            print('Loading VGG up to layer',deepest_target)
            self.vgg_model = vgg_from_t7(vgg_path, target_layer=deepest_target)
            print(self.vgg_model.summary())

        if self.mode == 'train':
            style_encodings = [None]  # Style encoding is not needed for train stage
        else:
            # Build model to extract intermediate relu layers for style img to be used in multi-level pipeline
            with tf.name_scope('style_encoder'):
                style_encoding_layers = [self.vgg_model.get_layer(relu).output for relu in relu_targets]
                style_encoder_model = Model(inputs=self.vgg_model.input, outputs=style_encoding_layers)
                style_encodings = style_encoder_model(self.style_input)

            if len(relu_targets) == 1:
                style_encodings = [style_encodings]

        # Build enc/decs for each target relu and hook the out of each decoder up to subsequent encoder input
        for i, (relu, style_encoded) in enumerate(zip(relu_targets, style_encodings)):
            print('Building encoder/decoder for relu target',relu)
            
            if i == 0:
                # Input tensor will be a placeholder for the first encoder/decoder
                input_tensor = None
            else:
                # Input to intermediate levels is the output from previous decoder
                input_tensor = clip(self.encoder_decoders[-1].decoded)
            
            enc_dec = self.build_model(relu, input_tensor=input_tensor, style_encoded_tensor=style_encoded, **kwargs)
        
            self.encoder_decoders.append(enc_dec)

        # Hooks for placeholder input for first encoder and final output from last decoder
        self.content_input  = self.encoder_decoders[0].content_input
        self.decoded_output = self.encoder_decoders[-1].decoded
        
    def build_model(self, 
                    relu_target,
                    input_tensor,
                    style_encoded_tensor=None,
                    batch_size=8,
                    feature_weight=1,
                    pixel_weight=1,
                    tv_weight=0,
                    learning_rate=1e-4,
                    lr_decay=5e-5,
                    ss_patch_size=3,
                    ss_stride=1):
        '''Build the EncoderDecoder architecture for a given relu layer.

            Args:
                relu_target: Layer of VGG to decode from
                input_tensor: If None then a placeholder will be created, else use this tensor as the input to the encoder
                style_encoded_tensor: Tensor for style image features at the same relu layer. Used only at test time.
                batch_size: Batch size for training
                feature_weight: Float weight for feature reconstruction loss
                pixel_weight: Float weight for pixel reconstruction loss
                tv_weight: Float weight for total variation loss
                learning_rate: Float LR
                lr_decay: Float linear decay for training
            Returns:
                EncoderDecoder namedtuple with input/encoding/output tensors and ops for training.
        '''
        with tf.name_scope('encoder_decoder_'+relu_target):

            ### Build encoder for reluX_1
            with tf.name_scope('content_encoder_'+relu_target):
                if input_tensor is None:  
                    # This is the first level encoder that takes original content imgs
                    content_imgs = tf.placeholder_with_default(tf.constant([[[[0.,0.,0.]]]]), shape=(None, None, None, 3), name='content_imgs')
                else:                     
                    # This is an intermediate-level encoder that takes output tensor from previous level as input
                    content_imgs = input_tensor  

                # Build content layer encoding model
                content_layer = self.vgg_model.get_layer(relu_target).output
                content_encoder_model = Model(inputs=self.vgg_model.input, outputs=content_layer)

                # Setup content layer encodings for content images
                content_encoded = content_encoder_model(content_imgs)
 
            ### Build style encoder & WCT if test mode
            if self.mode != 'train':                
                with tf.name_scope('wct_'+relu_target):
                    if relu_target == 'relu5_1':
                        # Apply style swap on relu5_1 encodings if self.swap5 flag is set
                        # Use AdaIN as transfer op instead of WCT if self.use_adain is set
                        # Otherwise perform WCT
                        decoder_input = tf.case([(self.swap5, lambda: wct_style_swap(content_encoded,
                                                                                    style_encoded_tensor,
                                                                                    self.ss_alpha,
                                                                                    ss_patch_size, 
                                                                                    ss_stride)),
                                                (self.use_adain, lambda: adain(content_encoded, style_encoded_tensor, self.alpha))],
                                                default=lambda: wct_tf(content_encoded, style_encoded_tensor, self.alpha))
                    else:
                        decoder_input = tf.cond(self.use_adain, 
                                                lambda: adain(content_encoded, style_encoded_tensor, self.alpha),
                                                lambda: wct_tf(content_encoded, style_encoded_tensor, self.alpha))

                    
            else: # In train mode we're trying to reconstruct from the encoding, so pass along unchanged
                decoder_input = content_encoded

            ### Build decoder
            with tf.name_scope('decoder_'+relu_target):
                n_channels = content_encoded.get_shape()[-1].value
                decoder_model = self.build_decoder(input_shape=(None, None, n_channels), relu_target=relu_target)

                # Wrap the decoder_input tensor so that it has the proper shape for decoder_model
                decoder_input_wrapped = tf.placeholder_with_default(decoder_input, shape=[None,None,None,n_channels])

                # Reconstruct/decode from encoding
                decoded = decoder_model(Lambda(lambda x: x)(decoder_input_wrapped)) # Lambda converts TF tensor to Keras

            # Content layer encoding for stylized out
            decoded_encoded = content_encoder_model(decoded)

        if self.mode == 'train':  # Train & summary ops only needed for training phase
            ### Losses
            with tf.name_scope('losses_'+relu_target):
                # Feature loss between encodings of original & reconstructed
                feature_loss = feature_weight * mse(decoded_encoded, content_encoded)

                # Pixel reconstruction loss between decoded/reconstructed img and original
                pixel_loss = pixel_weight * mse(decoded, content_imgs)

                # Total Variation loss
                if tv_weight > 0:
                    tv_loss = tv_weight * tf.reduce_mean(tf.image.total_variation(decoded))
                else:
                    tv_loss = tf.constant(0.)

                total_loss = feature_loss + pixel_loss + tv_loss

            ### Training ops
            with tf.name_scope('train_'+relu_target):
                global_step = tf.Variable(0, name='global_step_train', trainable=False)
                # self.learning_rate = tf.train.exponential_decay(learning_rate, self.global_step, 100, 0.96, staircase=False)
                learning_rate = torch_decay(learning_rate, global_step, lr_decay)
                d_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999)

                # Only train decoder vars, encoder is frozen
                d_vars = [var for var in tf.trainable_variables() if 'decoder_'+relu_target in var.name]

                train_op = d_optimizer.minimize(total_loss, var_list=d_vars, global_step=global_step)

            ### Loss & image summaries
            with tf.name_scope('summary_'+relu_target):
                feature_loss_summary = tf.summary.scalar('feature_loss', feature_loss)
                pixel_loss_summary = tf.summary.scalar('pixel_loss', pixel_loss)
                tv_loss_summary = tf.summary.scalar('tv_loss', tv_loss)
                total_loss_summary = tf.summary.scalar('total_loss', total_loss)

                content_imgs_summary = tf.summary.image('content_imgs', content_imgs)
                decoded_images_summary = tf.summary.image('decoded_images', clip(decoded))
                
                for var in d_vars:
                    tf.summary.histogram(var.op.name, var)

                summary_op = tf.summary.merge_all()
        else:
            # For inference set unnneeded ops to None
            pixel_loss, feature_loss, tv_loss, total_loss, train_op, global_step, learning_rate, summary_op = [None]*8

        # Put it all together
        encoder_decoder = EncoderDecoder(content_input=content_imgs, 
                                         content_encoder_model=content_encoder_model,
                                         content_encoded=content_encoded,
                                         style_encoded=style_encoded_tensor,
                                         decoder_input=decoder_input,
                                         decoder_model=decoder_model,
                                         decoded=decoded,
                                         decoded_encoded=decoded_encoded,
                                         pixel_loss=pixel_loss,
                                         feature_loss=feature_loss,
                                         tv_loss=tv_loss,
                                         total_loss=total_loss,
                                         train_op=train_op,
                                         global_step=global_step,
                                         learning_rate=learning_rate,
                                         summary_op=summary_op)
        
        return encoder_decoder

    def build_decoder(self, input_shape, relu_target): 
        '''Build the decoder architecture that reconstructs from a given VGG relu layer.

            Args:
                input_shape: Tuple of input tensor shape, needed for channel dimension
                relu_target: Layer of VGG to decode from
        '''
        decoder_num = dict(zip(['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'], range(1,6)))[relu_target]

        # Dict specifying the layers for each decoder level. relu5_1 is the deepest decoder and will contain all layers
        decoder_archs = {
            5: [ #    layer    filts      HxW  / InC->OutC                                     
                (Conv2DReflect, 512),  # 16x16 / 512->512
                (UpSampling2D,),       # 16x16 -> 32x32
                (Conv2DReflect, 512),  # 32x32 / 512->512
                (Conv2DReflect, 512),  # 32x32 / 512->512
                (Conv2DReflect, 512)], # 32x32 / 512->512
            4: [
                (Conv2DReflect, 256),  # 32x32 / 512->256
                (UpSampling2D,),       # 32x32 -> 64x64
                (Conv2DReflect, 256),  # 64x64 / 256->256
                (Conv2DReflect, 256),  # 64x64 / 256->256
                (Conv2DReflect, 256)], # 64x64 / 256->256
            3: [
                (Conv2DReflect, 128),  # 64x64 / 256->128
                (UpSampling2D,),       # 64x64 -> 128x128
                (Conv2DReflect, 128)], # 128x128 / 128->128
            2: [
                (Conv2DReflect, 64),   # 128x128 / 128->64
                (UpSampling2D,)],      # 128x128 -> 256x256
            1: [
                (Conv2DReflect, 64)]   # 256x256 / 64->64
        }

        code = Input(shape=input_shape, name='decoder_input_'+relu_target)
        x = code

        ### Work backwards from deepest decoder # and build layer by layer
        decoders = reversed(range(1, decoder_num+1))
        count = 0        
        for d in decoders:
            for layer_tup in decoder_archs[d]:
                # Unique layer names are needed to ensure var naming consistency with multiple decoders in graph
                layer_name = '{}_{}'.format(relu_target, count)

                if layer_tup[0] == Conv2DReflect:
                    x = Conv2DReflect(layer_name, filters=layer_tup[1], kernel_size=3, padding='valid', activation='relu', name=layer_name)(x)
                elif layer_tup[0] == UpSampling2D:
                    x = UpSampling2D(name=layer_name)(x)
                
                count += 1

        layer_name = '{}_{}'.format(relu_target, count) 
        output = Conv2DReflect(layer_name, filters=3, kernel_size=3, padding='valid', activation=None, name=layer_name)(x)  # 256x256 / 64->3
        
        decoder_model = Model(code, output, name='decoder_model_'+relu_target)
        
        print(decoder_model.summary())
        
        return decoder_model
