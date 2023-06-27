add_executable(example_segmentation src/main.cpp src/util.cpp src/seg.cpp src/watershed.cpp)
set_target_properties(example_segmentation PROPERTIES UNITY_BUILD_MODE BATCH UNITY_BUILD_BATCH_SIZE 10)
target_link_libraries(example_segmentation pipeline)
add_dependencies(example_segmentation pipeline)

