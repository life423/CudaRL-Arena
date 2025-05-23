cmake_minimum_required(VERSION 3.20)
project(cuda_project LANGUAGES CXX CUDA)

# build our proof-of-concept executable from src/main.cpp
add_executable(poc_app src/main.cpp)
