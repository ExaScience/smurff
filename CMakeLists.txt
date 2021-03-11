cmake_minimum_required (VERSION 3.5.0)

project (Smurff)

MESSAGE("smurff")

# OPTIONS

OPTION(ENABLE_BLAS "Look for any BLAS" ON)
OPTION(ENABLE_OPENBLAS "Look only for OPENBLAS library" OFF)
OPTION(ENABLE_MKL "Look only for MKL library" OFF)

OPTION(ENABLE_PROFILING "Enable performance counters" OFF)
OPTION(ENABLE_ASAN "Enable address sanitizer" OFF)
OPTION(ENABLE_TSAN "Enable thread sanitizer" OFF)

OPTION(ENABLE_VERBOSE_COMPILER_LOG "Enable verbose compilation log" OFF)
OPTION(ENABLE_VERBOSE_LINKER_LOG "Enable verbose linker log" OFF)

OPTION(ENABLE_BOOST "Enable usage of Boost for program options" ON)

OPTION(ENABLE_CMDLINE "Enable Command-Line Tools" ON)
OPTION(ENABLE_TESTS "Enable C++ Unit Tests" ON)
OPTION(ENABLE_PYTHON "Enable Python Wrapper" ON)
OPTION(ENABLE_MPI "Enable MPI Support" OFF)

SET(SMURFF_FLOAT_TYPE "double" CACHE STRING "Main compute type")

# INIT CMAKE

message("Initializing cmake ...")
MESSAGE( STATUS "CMAKE_VERSION: " ${CMAKE_VERSION} )
# the short system name, e.g. "Linux", "FreeBSD" or "Windows"
MESSAGE( STATUS "CMAKE_SYSTEM_NAME: " ${CMAKE_SYSTEM_NAME} )
# only the version part of CMAKE_SYSTEM
MESSAGE( STATUS "CMAKE_SYSTEM_VERSION: " ${CMAKE_SYSTEM_VERSION} )
# the processor name (e.g. "Intel(R) Pentium(R) M processor 2.00GHz")
MESSAGE( STATUS "CMAKE_SYSTEM_PROCESSOR: " ${CMAKE_SYSTEM_PROCESSOR} )


# Add new build types
SET(CMAKE_CXX_FLAGS_RELNOOPENMP
    "${GCC_RELEASE_FLAGS}"
    CACHE STRING "Flags used by the C++ compiler for release-without-OpenMP builds."
    )
SET(CMAKE_C_FLAGS_RELNOOPENMP
    "${GCC_RELEASE_FLAGS}"
    CACHE STRING "Flags used by the C compiler for release-without-OpenMP builds."
    )
SET(CMAKE_EXE_LINKER_FLAGS_RELNOOPENMP
    ""
    CACHE STRING "Flags used for linking binaries for release-without-OpenMP builds."
    )
SET(CMAKE_SHARED_LINKER_FLAGS_RELNOOPENMP
    ""
    CACHE STRING "Flags used by the shared libraries for release-without-OpenMP builds."
    )
MARK_AS_ADVANCED(
    CMAKE_CXX_FLAGS_RELNOOPENMP
    CMAKE_C_FLAGS_RELNOOPENMP
    CMAKE_EXE_LINKER_FLAGS_RELNOOPENMP
    CMAKE_SHARED_LINKER_FLAGS_RELNOOPENMP )

# FETCH OS

message("Fetching os and compiler ...")

#Fetch OS state and mixture thereof, like cygwin(unix+windows), OSX(unix+apple)
if (UNIX)
  MESSAGE( STATUS "OS: Unix system")
endif()
if (WIN32)
  MESSAGE( STATUS "OS: Windows system" )
endif()
if (APPLE)
  MESSAGE( STATUS "OS: Apple system")
endif()
if (MINGW)
  MESSAGE( STATUS "Compiler is a MINGW variant")
endif()
if (CYGWIN)
  MESSAGE( STATUS "Compiler is a CYGWIN variant")
endif()
if (BORLAND)
  MESSAGE( STATUS "Compiler is a BORLAND variant")
endif()
if (CMAKE_COMPILER_IS_GNUCC)
  MESSAGE( STATUS "Compiler is a gcc variant")
endif()

# FETCH COMPILER

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# needed for PyBIND11
# https://pybind11.readthedocs.io/en/stable/faq.html#someclass-declared-with-greater-visibility-than-the-type-of-its-field-someclass-member-wattributes
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

if (MSVC OR MSVC_IDE OR MSVC_VERSION OR MSVC60 OR MSVC70 OR MSVC71 OR MSVC80 OR MSVC90 OR MSVC10 OR MSVC11 OR MSVC12)
   message( STATUS "Compiler is a Microsoft variant")
   message( STATUS "MSVC: " ${MSVC} )
   message( STATUS "MSVC_IDE: " ${MSVC_IDE} )
   message( STATUS "MSVC_VERSION: " ${MSVC_VERSION} )
   message( STATUS "MSVC60: " ${MSVC60} )
   message( STATUS "MSVC70: " ${MSVC70} )
   message( STATUS "MSVC71: " ${MSVC71} )
   message( STATUS "MSVC80: " ${MSVC80} )
   message( STATUS "MSVC90: " ${MSVC90} )
   message( STATUS "MSVC10: " ${MSVC10} )
   message( STATUS "MSVC11: " ${MSVC11} )
   message( STATUS "MSVC12: " ${MSVC12} )

   add_definitions (/W3)
   add_definitions (/w44267 /w44244)

   #hide some irritating c++ unsafe function warnings
   add_definitions(-D_SCL_SECURE_NO_WARNINGS)
   add_definitions(-D_CRT_SECURE_NO_WARNINGS)
   
   add_definitions(-D_WINDOWS)

   set (CMAKE_EXE_LINKER_FLAGS "/SAFESEH:NO")

elseif (CMAKE_COMPILER_IS_GNUCXX OR APPLE)
   if(APPLE)
      message(STATUS "   -- OSX: System root directory found at: ${CMAKE_OSX_SYSROOT}")
      message(STATUS "   -- OSX: c compiler found at: ${CMAKE_C_COMPILER}")
      message(STATUS "   -- OSX: c++ compiler found at: ${CMAKE_CXX_COMPILER}")

      add_definitions(-DMAC_OSX)
   endif()

   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unknown-pragmas -Wnon-virtual-dtor -Wno-parentheses -fPIC")
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wno-unknown-pragmas -fPIC")

   # suppress some pesky Eigen 3.3.7 warnings
   if (CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 7.0)
     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-int-in-bool-context")
   endif()

   if (ENABLE_ASAN)
      add_compile_options(-fno-omit-frame-pointer -fsanitize=address)
      add_link_options(-fno-omit-frame-pointer -fsanitize=address)
   endif()
   
   if (ENABLE_TSAN)
      add_compile_options(-fno-omit-frame-pointer -fsanitize=thread)
      add_link_options(-fno-omit-frame-pointer -fsanitize=thread)
   endif()

else()
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unknown-pragmas -Wnon-virtual-dtor")
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wno-unknown-pragmas")
endif()

# INIT VERSIONING

message("Initializing versioning...")

#get GIT commit count we can later use as patch level
execute_process(
  COMMAND git rev-list HEAD --count
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  RESULT_VARIABLE GIT_COMMIT_COUNT_ERROR
  OUTPUT_VARIABLE GIT_COMMIT_COUNT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT "${GIT_COMMIT_COUNT_ERROR}" STREQUAL "0")
   set(GIT_COMMIT_COUNT "Unknown")
endif()

message(STATUS "GIT commit count ${GIT_COMMIT_COUNT}")

#get GIT commit SHA
execute_process(
  COMMAND git rev-parse HEAD
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  RESULT_VARIABLE GIT_COMMIT_ERROR
  OUTPUT_VARIABLE GIT_COMMIT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT "${GIT_COMMIT_ERROR}" STREQUAL "0")
   set(GIT_COMMIT "Unknown")
endif()

message(STATUS "GIT commit SHA ${GIT_COMMIT}")

#get GIT branch
execute_process(
  COMMAND git name-rev --name-only HEAD
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  RESULT_VARIABLE GIT_BRANCH_ERROR
  OUTPUT_VARIABLE GIT_BRANCH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT "${GIT_BRANCH_ERROR}" STREQUAL "0")
   set(GIT_BRANCH "Unknown")
endif()

message(STATUS "GIT branch ${GIT_BRANCH}")

#get GIT describe
execute_process(
  COMMAND git describe
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  RESULT_VARIABLE SMURFF_VERSION_ERROR
  OUTPUT_VARIABLE SMURFF_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT "${SMURFF_VERSION_ERROR}" STREQUAL "0")
   set(SMURFF_VERSION "Unknown")
endif()

message(STATUS "GIT describe ${SMURFF_VERSION}")

string(REGEX REPLACE "-[^-]+$"
    "" SMURFF_VERSION_CLEAN
    ${SMURFF_VERSION})

add_definitions(-DSMURFF_VERSION=\"${SMURFF_VERSION}\")

message(STATUS "Cleaned version: ${SMURFF_VERSION_CLEAN}")

# CONFIGURE BUILD TYPE

message("Analyzing build type...")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif()

if(NOT MSVC)
  if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Debug build.")
    add_definitions(-D_DEBUG)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
  elseif(CMAKE_BUILD_TYPE MATCHES Release)
    message(STATUS "Release build.")
  else()
    message(STATUS "Unknown build type...")
  endif()
endif()

# CONFIGURE LOGGING

message(STATUS "Initializing logging options...")

if (UNIX)
    #set verbose output of cmake
    if (${ENABLE_VERBOSE_COMPILER_LOG})
        set(CMAKE_VERBOSE_MAKEFILE ON)
    endif()

    #enable make verbose linker log
    if (${ENABLE_VERBOSE_LINKER_LOG})
        SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -###" )
    endif()
endif()

# CONFIGURE DEPENDENCIES

message("Checking build dependencies...")

#set additional search paths for cmake scripts

add_definitions(-DSMURFF_FLOAT_TYPE=${SMURFF_FLOAT_TYPE})
message(STATUS "Using '${SMURFF_FLOAT_TYPE}' as main computing type")
include(cmake/DependenciesConfig.cmake)

configure_eigen()
configure_highfive()
configure_boost()
configure_openmp()

if(ENABLE_MPI)
  configure_mpi()
endif()

set(ALGEBRA_LIB_FOUND OFF)

if(${ENABLE_OPENBLAS})
  if(${ENABLE_MKL})
    message(FATAL_ERROR "You cannot both enable OpenBLAS and MKL")
  endif()
  configure_openblas()
  set(ALGEBRA_LIBS ${BLAS_LIBRARIES})
  set(ALGEBRA_LIB_NAME OPENBLAS)
elseif(ENABLE_MKL)
  configure_mkl()
  set(ALGEBRA_LIBS ${MKL_LIBRARIES})
  set(ALGEBRA_LIB_NAME MKL)
elseif(ENABLE_BLAS)
  configure_lapack()
  set(ALGEBRA_LIBS ${LAPACKE_LIBRARIES} ${LAPACK_LIBRARIES})
  set(ALGEBRA_LIB_NAME LAPACK)
endif()

message(STATUS "Algebra lib name:  ${ALGEBRA_LIB_NAME}")
message(STATUS "Algebra libs: ${ALGEBRA_LIBS}")

if(${ENABLE_PROFILING})
    add_definitions(-DPROFILING)
endif()


# support for running "make test" (or alike)

enable_testing()

# unittests
add_test( unittests tests )

if(${EXTERNAL_DATA})
    file(DOWNLOAD  http://homes.esat.kuleuven.be/~jsimm/chembl-IC50-346targets.mm 
            ${CMAKE_CURRENT_BINARY_DIR}/jsimm-data/chembl-IC50-346targets.mm
            EXPECTED_HASH SHA256=10c3e1f989a7a415a585a175ed59eeaa33eff66272d47580374f26342cddaa88
        )

    file(DOWNLOAD http://homes.esat.kuleuven.be/~jsimm/chembl-IC50-compound-feat.mm
            ${CMAKE_CURRENT_BINARY_DIR}/jsimm-data/chembl-IC50-compound-feat.mm
            EXPECTED_HASH SHA256=f9fe0d296272ef26872409be6991200dbf4884b0cf6c96af8892abfd2b55e3bc
        )

    file(DOWNLOAD http://homes.esat.kuleuven.be/~jsimm/chembl-IC50-compounds.csv
            ${CMAKE_CURRENT_BINARY_DIR}/jsimm-data/chembl-IC50-compounds.csv
            EXPECTED_HASH SHA256=e8f045a67ee149c6100684e07920036de72583366596eb5748a79be6e3b96f7c
        )

    file(DOWNLOAD http://homes.esat.kuleuven.be/~jsimm/chembl-IC50-proteins-uniprot.csv
            ${CMAKE_CURRENT_BINARY_DIR}/jsimm-data/chembl-IC50-proteins-uniprot.csv
            EXPECTED_HASH SHA256=224b1b44abcab8448b023874f4676af30d64fe651754144f9cbdc67853b76ea8
        )

    add_test(chembl_bmpf smurff --num-latent=4 --burnin=2 --nsamples=2
        --train ${CMAKE_CURRENT_BINARY_DIR}/jsimm-data/chembl-IC50-346targets.mm
        )

    add_test(chembl_macau smurff --num-latent=4 --burnin=2 --nsamples=2
        --train ${CMAKE_CURRENT_BINARY_DIR}/jsimm-data/chembl-IC50-346targets.mm
        --row-features ${CMAKE_CURRENT_BINARY_DIR}/jsimm-data/chembl-IC50-compound-feat.mm
        )
endif()

include_directories(cpp)

# CONFIGURE TARGETS

message(STATUS "Creating folder structure...")

SET(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin")

# libraries
add_subdirectory (cpp/SmurffCpp)

# main executable
if(ENABLE_CMDLINE)
  add_subdirectory (cpp/Smurff)
endif()

# MPI
if(ENABLE_MPI)
  add_subdirectory (cpp/SmurffMPI)
endif()

# tests
if(ENABLE_TESTS)
  add_subdirectory (cpp/Tests)
endif()

# python
if(ENABLE_PYTHON)
    configure_python()
    add_subdirectory (python)
endif()

