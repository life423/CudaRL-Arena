# Python Bindings (pybind11)

pybind11_add_module(cuda_gridworld_bindings
    # Add your binding source files here, e.g.:
    # bindings.cpp
)

target_link_libraries(cuda_gridworld_bindings
    PRIVATE
        ai_cuda
        game_engine
)
