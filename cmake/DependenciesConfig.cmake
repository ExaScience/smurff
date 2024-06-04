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

macro(configure_blas)
  message ("Dependency check for BLAS/LAPACK/LAPACKE...")

  find_package(BLAS)
  if (BLAS_FOUND)
    message(STATUS "BLAS libraries: ${BLAS_LIBRARIES}" )
    message(STATUS "BLAS include: ${BLAS_INCLUDE_DIR}" )
    add_definitions(-DEIGEN_USE_BLAS)

    if (BLA_VENDOR MATCHES "Intel10")
      add_definitions(-DEIGEN_USE_MKL_ALL)
      message(STATUS "MKL Found" )
    endif()

    list(APPEND ALGEBRA_LIBS ${BLAS_LIBRARIES})

macro(configure_openblas)
  message ("Dependency check for openblas...")

  if(MSVC)
  set(BLAS_LIBRARIES  $ENV{BLAS_LIBRARIES})
  set(BLAS_INCLUDES $ENV{BLAS_INCLUDES})
  set(BLAS_FOUND ON)
  else()
    message(STATUS "BLAS NOT found" )
  endif (BLAS_FOUND)

  find_package(LAPACK)
  if (LAPACK_FOUND)
    message(STATUS "LAPACK libraries: ${LAPACK_LIBRARIES}" )
    message(STATUS "LAPACK include: ${LAPACK_INCLUDE_DIR}" )
    add_definitions(-DEIGEN_USE_LAPACK)

  message(STATUS BLAS: ${BLAS_LIBRARIES} )

endmacro(configure_openblas)

  find_package(LAPACKE)
  if (LAPACKE_FOUND)
    message(STATUS "LAPACKE libraries: ${LAPACKE_LIBRARIES}" )
    message(STATUS "LAPACKE include: ${LAPACKE_INCLUDE_DIR}" )
    add_definitions(-DEIGEN_USE_LAPACKE)
    list(APPEND ALGEBRA_LIBS ${LAPACKE_LIBRARIES})
  else()
    message(STATUS "LAPACKE NOT found" )
  endif(LAPACKE_FOUND)

  message(STATUS "all algebra libraries: ${ALGEBRA_LIBS}" )

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
      set(PYBIND11_NEWPYTHON ON)
        find_package(pybind11 CONFIG REQUIRED)
    endif()
endmacro(configure_python)
