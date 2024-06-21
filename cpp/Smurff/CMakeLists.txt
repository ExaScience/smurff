FILE (GLOB SOURCE_FILES "smurff.cpp")

source_group ("Source Files" FILES ${SOURCE_FILES})

#SETUP OUTPUT
add_executable (smurff ${HEADER_FILES} ${SOURCE_FILES})
install(TARGETS smurff RUNTIME DESTINATION bin)
target_link_libraries (smurff smurff-cpp ${Boost_LIBRARIES} ${ALGEBRA_LIBS} HighFive ${OpenMP_CXX_LIBRARIES})

if (ENABLE_TESTS)
    target_link_libraries(smurff Catch2::Catch2)
endif (ENABLE_TESTS)
