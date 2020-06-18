from libc.stdint cimport *
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr
from libcpp cimport bool

from NoiseConfig cimport NoiseConfig
from PVec cimport PVec

cdef extern from "<SmurffCpp/Configs/TensorConfig.h>" namespace "smurff":
    cdef cppclass TensorConfig:
        #
        # Sparse double tensor constructors
        #
        TensorConfig(shared_ptr[vector[uint64_t]] dims,
                     shared_ptr[vector[uint32_t]] columns, shared_ptr[vector[double]] values,
                     const NoiseConfig& noiseConfig, bool isScarce) except +

        #
        # Dense double tensor constructors
        #
        TensorConfig(shared_ptr[vector[uint64_t]] dims,
                     shared_ptr[vector[double]] values,
                     const NoiseConfig& noiseConfig) except +

        void setNoiseConfig(const NoiseConfig& value)
        shared_ptr[vector[uint64_t]] getDimsPtr();
        void setPos(const vector[int]& p);
