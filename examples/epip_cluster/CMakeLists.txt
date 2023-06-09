add_executable(tracker src/tracker.cpp src/tracker_main.cpp)
set_target_properties(tracker PROPERTIES UNITY_BUILD_MODE BATCH UNITY_BUILD_BATCH_SIZE 10)
target_link_libraries(tracker ${OpenCV_LIBS} ${FLANN_LIBRARY} ${Boost_LIBRARIES})
