cmake_minimum_required(VERSION 3.23)
project(galaxies CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(galaxies framework.cu)

set_target_properties(galaxies PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
