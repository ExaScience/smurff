FROM quay.io/pypa/manylinux2014_x86_64

RUN yum -y install wget eigen3-devel openblas-devel && \
   yum clean all

RUN wget https://cmake.org/files/v3.12/cmake-3.12.0-Linux-x86_64.sh && \
   sh cmake-3.12.0-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir && \
   rm -f cmake-3.12.0-Linux-x86_64.sh
