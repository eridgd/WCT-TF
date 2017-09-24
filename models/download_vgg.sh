# Borrowed from https://github.com/xunhuang1995/AdaIN-style/
# The VGG-19 network is obtained by:
# 1. converting vgg_normalised.caffemodel to .t7 using loadcaffe
# 2. inserting a convolutional module at the beginning to preprocess the image
# 3. replacing zero-padding with reflection-padding
# The original vgg_normalised.caffemodel can be obtained with:
# "wget -c --no-check-certificate https://bethgelab.org/media/uploads/deeptextures/vgg_normalised.caffemodel"
cd models
wget -c -O vgg_normalised.t7 "https://www.dropbox.com/s/kh8izr3fkvhitfn/vgg_normalised.t7?dl=1"
cd .. 