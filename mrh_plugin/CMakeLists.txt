cmake_minimum_required(VERSION 2.8.3)
project(simple_controller_plugin)

find_package(roscpp REQUIRED)
include_directories(/user/include/sdformat-9.10)
include_directories(./include)
find_package(gazebo REQUIRED)
include_directories(${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})
list(APPEND CMAKE_CXX_FLAGS "${GAZEBO_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GAZEBO_CXX_FLAGS}")

add_library(simple_controller_plugin SHARED src/simple_controller_plugin.cpp)
target_link_libraries(simple_controller_plugin ${GAZEBO_LIBRARIES} ${roscpp_LIBRARIES})
set_target_properties(simple_controller_plugin PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)