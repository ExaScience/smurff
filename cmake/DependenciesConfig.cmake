set (SCRIPT_DIR "${CMAKE_SOURCE_DIR}/cmake/")
set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

macro(configure_mpi)
  message ("Dependency check for mpi...")

  find_package(MPI)
  if(${MPI_C_FOUND})
    message(STATUS "MPI found")
  else()
    message(STATUS "MPI not found")
  endif()
   
endmacro(configure_mpi)

macro(configure_openmp)
  if (MSVC)
    message ("Skipped check for OpenMP (Windows)")
    set(OPENMP_FOUND FALSE)
  elseif(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    message ("Dependency check for OpenMP")

    find_package(OpenMP)
    if(${OPENMP_FOUND})
        message(STATUS "OpenMP found")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        include_directories(${OpenMP_CXX_INCLUDE_DIRS})

        message(STATUS "OpenMP_CXX_LIB_NAMES ${OpenMP_CXX_LIB_NAMES}")
        message(STATUS "OpenMP_CXX_LIBRARY ${OpenMP_CXX_LIBRARY}")
        message(STATUS "OpenMP_CXX_LIBRARIES ${OpenMP_CXX_LIBRARIES}")
        message(STATUS "OpenMP_CXX_INCLUDE_DIRS ${OpenMP_CXX_INCLUDE_DIRS}")
        message(STATUS "OpenMP_CXX_FLAGS ${OpenMP_CXX_FLAGS}")
    else()
        message(STATUS "OpenMP not found")
    endif()
  else()
    message ("Skipped check for OpenMP (Debug/NoOpenMP build)")
    set(OPENMP_FOUND FALSE)
  endif()   
endmacro(configure_openmp)

macro(configure_lapack)
  message ("Dependency check for lapack...")
  find_package(LAPACK REQUIRED)
  find_package(LAPACKE REQUIRED)
  add_definitions(-DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE)
  # needed because MSVC does not have support for c-type _Complex
  add_definitions(-Dlapack_complex_float=std::complex<float> -Dlapack_complex_double=std::complex<double>)
  message(STATUS LAPACK: ${LAPACK_LIBRARIES})
endmacro(configure_lapack)

macro(configure_openblas)
  message ("Dependency check for openblas...")
  
  if(MSVC)
  set(BLAS_LIBRARIES  $ENV{BLAS_LIBRARIES})
  set(BLAS_INCLUDES $ENV{BLAS_INCLUDES})
  set(BLAS_FOUND ON)
  else()
  set(BLA_VENDOR "OpenBLAS")
  find_package( BLAS REQUIRED )
  endif()

  add_definitions(-DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE)

  message(STATUS BLAS: ${BLAS_LIBRARIES} )
 
endmacro(configure_openblas)

macro(configure_mkl)
  message ("Dependency check for MKL (using MKL SDL)...")
  find_library (MKL_LIBRARIES "mkl_rt" HINTS ENV LD_LIBRARY_PATH REQUIRED)
  find_PATH (MKL_INCLUDE_DIR "mkl.h" HINTS ENV CPATH REQUIRED)

  include_directories(${MKL_INCLUDE_DIR})

  add_definitions(-DEIGEN_USE_MKL_ALL)
  
  message(STATUS "MKL libraries: ${MKL_LIBRARIES}" )
  message(STATUS "MKL include: ${MKL_INCLUDE_DIR}" )
endmacro(configure_mkl)

macro(configure_eigen)
  message ("Dependency check for eigen...")
  
  if(DEFINED ENV{EIGEN3_INCLUDE_DIR})
    SET(EIGEN3_INCLUDE_DIR $ENV{EIGEN3_INCLUDE_DIR})
  else()
    find_package(Eigen3 REQUIRED)
  endif()

  include_directories(${EIGEN3_INCLUDE_DIR})
  message(STATUS EIGEN3: ${EIGEN3_INCLUDE_DIR})
endmacro(configure_eigen)

macro(configure_highfive)
  message ("Dependency check for HighFive...")
  SET(HIGHFIVE_USE_BOOST OFF CACHE BOOL "Disable BOOST support in HighFive")
  find_package(HighFive REQUIRED)
endmacro(configure_highfive)

macro(configure_boost)
  message ("Dependency check for boost...")
  if(${ENABLE_BOOST})
      FIND_PACKAGE(Boost COMPONENTS system program_options filesystem REQUIRED)

      message("-- Found Boost_VERSION: ${Boost_VERSION}")
      message("-- Found Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
      message("-- Found Boost_LIBRARY_DIRS: ${Boost_LIBRARY_DIRS}")
      message("-- Found Boost_LIBRARIES: ${Boost_LIBRARIES}")
      add_definitions(-DHAVE_BOOST -DBOOST_ALL_NO_LIB)

      include_directories(${Boost_INCLUDE_DIRS})
  else()
      message("-- Boost library is not enabled")
  endif()
endmacro(configure_boost)

macro(configure_python)
    if(ENABLE_PYTHON)
        find_package(pybind11 CONFIG REQUIRED)
    endif()
endmacro(configure_python)
