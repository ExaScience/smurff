pybind11_add_module(wrapper src/smurff/wrapper.cpp)

#LINK LIBRARIES
target_link_libraries (wrapper PRIVATE smurff-cpp pybind11::module
                            ${ALGEBRA_LIBS}
                            HighFive
                            ${OpenMP_CXX_LIBRARIES}

                     )
