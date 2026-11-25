cmake_minimum_required(VERSION 3.24)
project(PondFlexibleTailCheck LANGUAGES CXX)

add_executable(pond_flexible_tail_check check.cpp)
target_compile_features(pond_flexible_tail_check PRIVATE cxx_std_17)
target_include_directories(pond_flexible_tail_check PRIVATE ../../src ../../src/nn)

# Both the standalone Godot project and this native fixture read one habitat
# descriptor, with dimensions in metres.
set(habitat_path "${CMAKE_CURRENT_LIST_DIR}/../../flexible-tail-preview/habitat.json")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${habitat_path}")
file(READ "${habitat_path}" habitat)
string(JSON habitat_schema GET "${habitat}" schema)
if(NOT habitat_schema EQUAL 1)
    message(FATAL_ERROR "Unsupported flexible-tail habitat descriptor")
endif()
foreach(dimension IN ITEMS water_radius_x_m water_radius_z_m flat_radius_x_m
                           flat_radius_z_m floor_y_m reference_scale_m)
    string(JSON ${dimension} GET "${habitat}" ${dimension})
endforeach()
configure_file(habitat_config.h.in habitat_config.h @ONLY)
target_include_directories(pond_flexible_tail_check PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
add_executable(pond_flexible_tail_choice_check choice-check.cpp)
target_compile_features(pond_flexible_tail_choice_check PRIVATE cxx_std_17)
target_include_directories(pond_flexible_tail_choice_check PRIVATE ../../src)
if(MSVC)
    target_compile_options(pond_flexible_tail_check PRIVATE /W4)
    target_compile_options(pond_flexible_tail_choice_check PRIVATE /W4)
endif()
