add_library(cudarl_core
    environment.cpp
    vectorized_environment.cpp
)
target_include_directories(cudarl_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
