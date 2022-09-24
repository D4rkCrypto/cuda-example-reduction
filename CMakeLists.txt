cmake_minimum_required(VERSION 3.23)
project(reduction-example CUDA)

set(CMAKE_CUDA_STANDARD 17)

# Boost
find_package(Boost 1.78 REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

add_executable(reduction reduction.cu)
