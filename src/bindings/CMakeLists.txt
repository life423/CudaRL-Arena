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

#include <SDL.h>
#include <iostream>

#int main() {
#    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
#        std::cerr << "SDL_Init Error: " << SDL_GetError() << std::endl;
#        return 1;
#    }
#    SDL_Window* win = SDL_CreateWindow("SDL2 Test", 100, 100, 640, 480, SDL_WINDOW_SHOWN);
#    if (win == nullptr) {
#        std::cerr << "SDL_CreateWindow Error: " << SDL_GetError() << std::endl;
#        SDL_Quit();
#        return 1;
#    }
#    SDL_Delay(1000); // Show window for 1 second
#    SDL_DestroyWindow(win);
#    SDL_Quit();
#    std::cout << "SDL2 is working correctly!" << std::endl;
#    return 0;
#}

add_executable(sdl2_test sdl2_test.cpp)
target_link_libraries(sdl2_test PRIVATE ${SDL2_LIBRARIES})
target_include_directories(sdl2_test PRIVATE ${SDL2_INCLUDE_DIRS})