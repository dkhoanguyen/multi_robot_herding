cmake_minimum_required(VERSION 2.8.3)
project(mrh_robot)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    std_msgs
    sensor_msgs
    geometry_msgs
    nav_msgs
    tf
    tf2
    gazebo_ros
)
find_package(GTest REQUIRED)

find_package(gazebo REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS roscpp std_msgs sensor_msgs geometry_msgs nav_msgs tf tf2 gazebo_ros std_msgs
  DEPENDS gazebo
)

# Set the path to the Eigen library
set(EIGEN_DIR "/usr/include/eigen3")

# Include the Eigen headers
include_directories(${EIGEN_DIR})
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

set(robot_behavior_src
    src/herding_by_surrounding.cpp)
add_executable(${PROJECT_NAME} ${robot_behavior_src})
add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES})

# include_directories(${GTEST_INCLUDE_DIRS})
# catkin_add_gtest(test_herding_by_surrounding
#     test/test_herding_by_surrounding.cpp)
# target_link_libraries(test_herding_by_surrounding
#     ${catkin_LIBRARIES}
#     ${GTEST_LIBRARIES}
#     ${PROJECT_NAME}
# )
# catkin_add_gtest(test_saber_flocking
#     test/test_saber_flocking.cpp)
# target_link_libraries(test_saber_flocking
#     ${catkin_LIBRARIES}
#     ${GTEST_LIBRARIES}
#     behavior_list
# )
