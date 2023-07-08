cmake_minimum_required(VERSION 2.8.3)
project(mrh_animal)

find_package(roscpp REQUIRED)
include_directories(/user/include/sdformat-9.10)
include_directories(./include)
find_package(gazebo REQUIRED)
include_directories(${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})
list(APPEND CMAKE_CXX_FLAGS "${GAZEBO_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GAZEBO_CXX_FLAGS}")

set(behavior_lib_src
    src/wandering_behavior.cpp)
add_library(behavior_list SHARED ${behavior_lib_src})
target_link_libraries(behavior_list ${GAZEBO_LIBRARIES} ${roscpp_LIBRARIES})

add_library(animal_behavior_plugin SHARED src/animal_behavior_plugin.cpp)
target_link_libraries(animal_behavior_plugin ${GAZEBO_LIBRARIES} ${roscpp_LIBRARIES} behavior_list)
set_target_properties(animal_behavior_plugin PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)