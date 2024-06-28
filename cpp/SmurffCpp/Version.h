#pragma once

#include <SmurffCpp/Utils/Error.h>

#ifdef ENABLE_BOOST
#include <boost/version.hpp>
#endif

#include <Eigen/Core>
#include <highfive/H5Version.hpp>

namespace smurff {
   inline std::string full_version() {
      std::string v(SMURFF_VERSION);

#ifdef HAVE_BOOST
      v += " (Boost version: " BOOST_LIB_VERSION ")";
#else
      v += " (no Boost)";
#endif

      v += " (Eigen " STRINGIFY(EIGEN_WORLD_VERSION)
                  "." STRINGIFY(EIGEN_MAJOR_VERSION)
                  "." STRINGIFY(EIGEN_MINOR_VERSION)
             ")";

#ifdef BLA_VENDOR
      v += " (BLAS/LAPACK vendor: " STRINGIFY(BLA_VENDOR) ")";
#elif defined(EIGEN_USE_BLAS)
      v += " (generic BLAS)";
#else
      v += " (no BLAS)";
#endif

      v += " (HighFive " STRINGIFY(HIGHFIVE_VERSION) ")";


#if defined(_OPENMP)
      v += " (OpenMP)";
#else
      v += " (single threaded)";
#endif

      return v;
   }

};

#include <SmurffCpp/Utils/Tensor.h>