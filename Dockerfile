FROM tensorflow/tensorflow:1.2.1-gpu-py3

RUN sed -i "s/jessie main/jessie main contrib non-free/" /etc/apt/sources.list
RUN echo "deb http://http.debian.net/debian jessie-backports main contrib non-free" >> /etc/apt/sources.list

RUN gpg --keyserver pgpkeys.mit.edu --recv-key 7638D0442B90D010     
RUN gpg -a --export 7638D0442B90D010 | apt-key add -

RUN apt-get update && \
    apt-get install -y \
        build-essential \
        cmake \
        ffmpeg \
        gfortran \
        git \
        libatlas-base-dev \
        libavcodec-dev \
        libavformat-dev \
        libcanberra-gtk-module \
        libgtk2.0-dev \
        libjasper-dev \
        libjpeg-dev \
        libpng-dev \
        libpng12-dev \
        libpq-dev \
        libswscale-dev \
        libtbb-dev \
        libtbb2 \
        libtiff-dev \
        libtiff5-dev \
        libv4l-dev \
        libx264-dev \
        libxvidcore-dev \
        pkg-config \
        python2.7-dev \
        python3.5-dev \
        python-pip \
        unzip \
        wget \
        yasm \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install numpy
RUN pip2 install numpy

WORKDIR /
RUN wget https://github.com/opencv/opencv_contrib/archive/3.2.0.zip \
&& unzip 3.2.0.zip \
&& rm 3.2.0.zip

RUN wget https://github.com/Itseez/opencv/archive/3.2.0.zip \
&& unzip 3.2.0.zip \
&& mkdir /opencv-3.2.0/cmake_binary \
&& cd /opencv-3.2.0/cmake_binary \
&& cmake -DBUILD_TIFF=ON \
  -DBUILD_opencv_java=OFF \
  -DOPENCV_EXTRA_MODULES_PATH=/opencv_contrib-3.2.0/modules \
  -DWITH_CUDA=OFF \
  -DENABLE_AVX=ON \
  -DWITH_OPENGL=ON \
  -DWITH_OPENCL=ON \
  -DWITH_IPP=ON \
  -DWITH_TBB=ON \
  -DWITH_EIGEN=ON \
  -DWITH_V4L=ON \
  -DBUILD_TESTS=OFF \
  -DBUILD_PERF_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DCMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") \
  -DPYTHON_EXECUTABLE=$(which python3) \
  -DPYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
  -DPYTHON_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") .. \
&& make install \
&& rm /3.2.0.zip \
&& rm -r /opencv-3.2.0 \
&& rm -r /opencv_contrib-3.2.0

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD python3 webcam.py \
    --checkpoints models/relu5_1 models/relu4_1 models/relu3_1 models/relu2_1 models/relu1_1 \
    --relu-targets relu5_1 relu4_1 relu3_1 relu2_1 relu1_1 \
    --style-size 512 \
    --alpha 0.8 \
    --style-path images
