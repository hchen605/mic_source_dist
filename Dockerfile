FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04
## MAINTAINER Chun-wei Ho <chun-wei.ho@gatech.edu>

##INSTALL CMAKE SOX SNDFILE FFMPEG FLAC etc using apt-get install
RUN apt-get update --fix-missing && \
    apt-get install -y \
    g++ make cmake htop sox libsndfile1-dev ffmpeg flac git zlib1g-dev automake autoconf bzip2 unzip wget gfortran libtool git subversion python2.7 python3.8 python3-pip ca-certificates patch valgrind libssl-dev vim gawk libtiff5-dev libjpeg8-dev libopenjp2-7-dev libfreetype6-dev 

##INSTALL NVIDIA-DRIVER
RUN apt-get update && \
    apt-get install -y nvidia-utils-440

# Install conda
ENV CONDA_DIR /root/anaconda3
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh -P ~ && \
    bash ~/Anaconda3-2022.10-Linux-x86_64.sh -b
ENV PATH=$CONDA_DIR/bin:$PATH

# Install required packages
COPY mic.yaml .
RUN conda init
RUN conda env create --file mic.yaml && \
    ln -s /usr/local/cuda-11.1/lib64/libcusolver.so.11 /usr/local/cuda-11.1/lib64/libcusolver.so.10 && \
    echo >> ~/.bashrc && \
    echo "conda activate mic" >> ~/.bashrc  && \
    echo "export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc

# Update cuda path

# Install Pytorch
SHELL ["conda", "run", "-n", "mic", "/bin/bash", "-c"]
RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 pytorch-ignite msgpack-numpy \
    -f https://download.pytorch.org/whl/torch_stable.html


# Download model and move source code to the docker env
RUN mkdir -p /home/mic_source_dist/ESResNeXt-fbsp/assets && \
    wget --quiet -nc -O /home/mic_source_dist/ESResNeXt-fbsp/assets/ESResNeXtFBSP_AudioSet.pt \
    https://github.com/AndreyGuzhov/ESResNeXt-fbsp/releases/download/v0.1/ESResNeXtFBSP_AudioSet.pt
ADD . /home/mic_source_dist/
