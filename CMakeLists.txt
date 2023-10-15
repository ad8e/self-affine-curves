cmake_minimum_required(VERSION 3.5)

project(curvesolver)

find_package(Ceres REQUIRED)
cmake_minimum_required(VERSION 3.5)
set(EIGEN_INCLUDE_DIR "/usr/local/include/eigen3")
include_directories(${EIGEN_INCLUDE_DIR})

add_executable(tiny tiny_solver.cpp)

target_compile_features(tiny PRIVATE cxx_std_23)
target_include_directories(tiny PRIVATE "/home/a/stuff/vsync/j/")
target_link_libraries(tiny Ceres::ceres)
add_definitions(-DCMAKE_EXPORT_COMPILE_COMMANDS=ON)


add_executable(tinyhtml tiny_html_driver.cpp)
target_compile_features(tinyhtml PRIVATE cxx_std_23)
target_include_directories(tinyhtml PRIVATE "/home/a/stuff/vsync/j/")
target_link_libraries(tinyhtml Ceres::ceres)