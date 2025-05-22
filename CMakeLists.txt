cmake_minimum_required(VERSION 3.18)
project(VectorAdd CUDA)

set(CMAKE_CUDA_STANDARD 11)

add_executable(vector_add vector_add.cu)
