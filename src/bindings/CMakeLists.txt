# Python Bindings (pybind11)

pybind11_add_module(cuda_gridworld_bindings
    bindings.cpp
)

target_link_libraries(cuda_gridworld_bindings
    PRIVATE
        ai_cuda
        game_engine
)

# Ensure the Python module is installed to the correct location
set_target_properties(cuda_gridworld_bindings PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/interface"
)