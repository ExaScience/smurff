FROM ubuntu:19.10

RUN apt-get update && \
   DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       cmake \
       git g++ python3 \
       libblas-dev \
       liblapack-dev \
       liblapacke-dev \
       libopenmpi-dev \
       openmpi-bin \
       libeigen3-dev \
       libboost-all-dev \
       ca-certificates \
       libhdf5-dev \
       gdb vim wget\
       libhdf5-dev \
       python3-numpy python3-pybind11 python3-setuptools \
       python3-scipy python3-pandas \
       python3-joblib python3-sklearn \
       python3-h5py \
       python3-pytest python3-parameterized \
    && rm -rf /var/lib/apt/lists/*

RUN wget -O HighFive.tar.gz https://github.com/BlueBrain/HighFive/archive/v2.2.tar.gz && \
    tar xzf HighFive.tar.gz && \
    cd HighFive* && mkdir build && cd build && \
    cmake .. && \
    make -j2 && make install && \
    cd ../../ && rm -r HighFive*

