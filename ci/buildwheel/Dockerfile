FROM quay.io/pypa/manylinux2014_x86_64

RUN yum -y install wget eigen3-devel openblas-devel hdf5-devel && \
   yum clean all

RUN wget https://cmake.org/files/v3.12/cmake-3.12.0-Linux-x86_64.sh && \
   sh cmake-3.12.0-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir && \
   rm -f cmake-3.12.0-Linux-x86_64.sh

#install HighFive
RUN wget -O HighFive.tar.gz https://github.com/BlueBrain/HighFive/archive/v2.2.tar.gz && \
    tar xzf HighFive.tar.gz && \
    rm HighFive.tar.gz && \
    cd HighFive* && \
    mkdir build && \
    cd build && \
    cmake .. -DHIGHFIVE_USE_BOOST=OFF && \
    make -j2 && \
    make install && \
    cd ../.. && \
    rm -r HighFive*

#install pybind11
RUN wget -O pybind11.tar.gz https://github.com/pybind/pybind11/archive/v2.5.0.tar.gz && \
   tar xzf pybind11.tar.gz && \
   rm pybind11.tar.gz && \
   cd pybind11* && \
   mkdir build && \
   cd build && \
   cmake .. -DPYBIND11_TEST=OFF && \
   make install && \
   cd ../.. && \
   rm -r pybind11*
