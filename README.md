# Universal Style Transfer via Whiten-Color Transform in TensorFlow & Keras

This is a TensorFlow/Keras implementation of [Universal Style Transfer via Feature Transforms](https://arxiv.org/abs/1703.06868) by Li et al. The core architecture is an auto-encoder trained to reconstruct from intermediate layers of a pre-trained VGG19 image classification net. Stylization is accomplished by matching the statistics of content/style image features through the [Whiten-Color Transform (WCT)](https://www.projectrhea.org/rhea/index.php/ECE662_Whitening_and_Coloring_Transforms_S14_MH), which is implemented here in both TensorFlow and NumPy. No style images are used for training, and the WCT allows for 'universal' style transfer for arbitrary content/style image pairs.

As in the original paper, reconstruction decoders for layers `reluX_1 (X=1,2,3,4,5)` are trained separately and then hooked up in a multi-level stylization pipeline in a single graph. A single VGG encoder is shared by all decoders to reduce memory usage.

This repo is based on [my implementation](https://github.com/eridgd/AdaIN-TF/) of [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868).


## Requirements

* Python 3.x
* tensorflow 1.2.1+
* keras 2.0.x
* torchfile 

Optionally:

* OpenCV with contrib modules (for `webcam.py`)
  * MacOS install http://www.pyimagesearch.com/2016/12/05/macos-install-opencv-3-and-python-3-5/
  * Linux install http://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/
* ffmpeg (for video stylization)


## Running a pre-trained model

1. Download VGG19 model: `bash models/download_vgg.sh`

2. Download checkpoints for the five decoders: `bash models/download_models.sh`

3. Obtain style images, e.g. from the [Wikiart dataset](https://www.kaggle.com/c/painter-by-numbers)

4. Run stylization for live video with `webcam.py` or for images with `stylize.py`. Both scripts share the same required arguments. For instance, to run a multi-level stylization pipeline that goes from relu5_1->relu4_1->relu3_1->relu2_1->relu1_1:

`python webcam.py --checkpoints models/relu5_1 models/relu4_1 models/relu3_1 models/relu2_1 models/relu1_1 \
--relu-targets relu5_1 relu4_1 relu3_1 relu2_1 relu1_1 \
--style-size 512 --alpha 0.8 \
--style-path /path/to/styleimgs`

The args `--checkpoints` and `--relu-targets` specify space-delimited lists of decoder checkpoint folders and corresponding relu layer targets. The order of relu targets determines the stylization pipeline order, where the output of one encoder/decoder becomes the input for the next. Specifying one checkpoint/relu target will perform single-level stylization.

Other args to take note of are:

* `--style-path`  Folder of style images or a single style image 
* `--style-size`  Resize small side of style image to this
* `--crop-size`  If specified center-crop a square of this size from the (resized) style image
* `--alpha`  [0,1] blending of content features + whiten-color transformed features to control degree of stylization
* `--passes`  # of times to run the stylization pipeline
* `--source`  Specify camera input ID, default 0
* `--width` and `--height`  Set the size of camera frames
* `--video-out`  Write stylized frames to .mp4 out path
* `--fps`  Frames Per Second for video out
* `--scale`  Resize content images by this factor before stylizing
* `--keep-colors`  Apply CORAL transform to preserve colors of content
* `--device`  Device to perform compute on, default `/gpu:0`
* `--concat`  Append the style image to the stylized output
* `--interpolate`  Interpolate between AdaIN features of two random images
* `--noise`  Generate textures from random noise image instead of webcam
* `--random`  Load a new random image every # of frames

There are also four keyboard shortcuts:

* `r`  Load random image from style folder
* `s`  Save frame to a .png
* `c`  Toggle color preservation
* `q`  Quit cleanly and close streams

Additionally, `stylize.py` will stylize image files. The options are the same as for the webcam script with the addition of `--content-path`, which can be a single image file or folder, and `--out-path` to specify the output folder. Each style in `--style-path` will be applied to each content image. 


## Training

1. Download [MS COCO images](http://mscoco.org/dataset/#download).

2. Download VGG19 model: `bash models/download_vgg.sh`

3. Train one decoder per relu target layer. E.g. to train a decoder to reconstruct from relu3_1:

`python train.py --relu-target relu3_1 --content-path /path/to/coco --batch-size 8 --feature-weight 1 --pixel-weight 1 --tv-weight 0 --checkpoint /path/to/checkpointdir --learning-rate 1e-4 --maxiter 15000`

3. Monitor training with TensorBoard: `tensorboard --logdir /path/to/checkpointdir`


## Notes

* The stylization pipeline can be hooked up with decoders in any order. For instance, to reproduce the (sub-optimal) reversed fine-to-coarse pipeline in figure 5(d) from the original paper, use the option `--relu-targets relu1_1 relu2_1 relu3_1 relu4_1 relu5_1` in webcam.py/stylize.py. 
* `coral.py` implements [CORellation ALignment](https://arxiv.org/abs/1612.01939) to transfer colors from the content image to the style image in order to preserve colors in the stylized output. The default method uses numpy, and I have also translated the author's CORAL code from Torch to PyTorch.


## Acknowledgments

Many thanks to the authors Yijun Li & collaborators at Adobe for their work that inspired this fun project. After building the first version of this TF implementation I discovered their [official Torch implementation](https://github.com/Yijunmaverick/UniversalStyleTransfer) that I referred to in tweaking the WCT op to be more stable. 


## TODO

- [ ] Interpolation between styles
- [ ] Video stylizatino
- [ ] Webcam style window threading
- [ ] Forget this static graph nonsense and redo everything in PyTorch
