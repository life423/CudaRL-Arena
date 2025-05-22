# C++ Game Engine (Gridworld) - SDL2

add_library(game_engine STATIC
    gridworld.cpp
    procedural_gridworld.cpp
)

target_include_directories(game_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(game_engine PUBLIC ${SDL2_LIBRARIES})