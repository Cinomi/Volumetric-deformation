cmake_minimum_required(VERSION 2.8.12)
project(Deformation_without_Normal)

add_executable(${PROJECT_NAME}_bin main.cpp)
include_directories(/usr/local/include/eigen3)
target_link_libraries(${PROJECT_NAME}_bin igl::core igl::viewer tutorials)
