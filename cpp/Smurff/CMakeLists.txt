FILE (GLOB SOURCE_FILES "smurff.cpp")

source_group ("Source Files" FILES ${SOURCE_FILES})

#SETUP OUTPUT
add_executable (smurff ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries (smurff smurff-cpp ${Boost_LIBRARIES} ${ALGEBRA_LIBS} HighFive ${OpenMP_CXX_LIBRARIES})

if (SKBUILD)
  install(TARGETS smurff RUNTIME DESTINATION "${SKBUILD_SCRIPTS_DIR}")
else (SKBUILD)
  install(TARGETS smurff RUNTIME DESTINATION bin)
endif (SKBUILD)

if (ENABLE_TESTS)
    target_link_libraries(smurff tests Catch2::Catch2)
endif (ENABLE_TESTS)
