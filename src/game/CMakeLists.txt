# C++ Game Engine (Gridworld) - SDL2

add_library(game_engine STATIC
    # Add your game source files here, e.g.:
    # gridworld.cpp
)

target_include_directories(game_engine PUBLIC ${SDL2_INCLUDE_DIRS})
target_link_libraries(game_engine PUBLIC ${SDL2_LIBRARIES})
