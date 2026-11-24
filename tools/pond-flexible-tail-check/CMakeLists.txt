cmake_minimum_required(VERSION 3.24)
project(PondFlexibleTailCheck LANGUAGES CXX)

add_executable(pond_flexible_tail_check check.cpp)
target_compile_features(pond_flexible_tail_check PRIVATE cxx_std_17)
target_include_directories(pond_flexible_tail_check PRIVATE ../../src ../../src/nn)
if(MSVC)
    target_compile_options(pond_flexible_tail_check PRIVATE /W4)
endif()
