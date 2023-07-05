add_executable(example_kitti src/main.cpp)
set_target_properties(example_kitti PROPERTIES UNITY_BUILD_MODE BATCH UNITY_BUILD_BATCH_SIZE 10)
target_link_libraries(example_kitti pipeline)
add_dependencies(example_kitti pipeline)

