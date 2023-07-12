cmake_minimum_required(VERSION 2.8.3)
project(mrh_robot)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    sensor_msgs
    geometry_msgs
    nav_msgs
    tf
    tf2
    gazebo_ros
)
find_package(GTest REQUIRED)

include_directories(./include)
find_package(gazebo REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp std_msgs sensor_msgs geometry_msgs nav_msgs tf tf2 gazebo_ros
  DEPENDS gazebo
)

# # Set the path to the Eigen library
# set(EIGEN_DIR "/usr/include/eigen3")

# # Include the Eigen headers
# include_directories(${EIGEN_DIR})

# set(behavior_lib_src
#     src/saber_flocking.cpp)
# add_library(behavior_list SHARED ${behavior_lib_src})
# target_link_libraries(behavior_list ${GAZEBO_LIBRARIES} ${catkin_LIBRARIES})

# add_library(animal_behavior_plugin SHARED src/animal_behavior_plugin.cpp)
# target_link_libraries(animal_behavior_plugin ${GAZEBO_LIBRARIES} ${catkin_LIBRARIES} behavior_list)
# set_target_properties(animal_behavior_plugin PROPERTIES
#     LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

# include_directories(${GTEST_INCLUDE_DIRS})
# catkin_add_gtest(test_math_utils
#     test/test_math_utils.cpp)
# target_link_libraries(test_math_utils
#     ${catkin_LIBRARIES}
#     ${GTEST_LIBRARIES}
# )
# catkin_add_gtest(test_saber_flocking
#     test/test_saber_flocking.cpp)
# target_link_libraries(test_saber_flocking
#     ${catkin_LIBRARIES}
#     ${GTEST_LIBRARIES}
#     behavior_list
# )
