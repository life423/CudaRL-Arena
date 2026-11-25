cmake_minimum_required(VERSION 3.24)
project(PondFlexibleTailLive LANGUAGES CXX)

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
get_filename_component(repo "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(CUDARL_GODOT_CPP_ROOT "${repo}/external/godot-cpp" CACHE PATH "Godot 4.4 C++ bindings")
set(godot_cpp_root "${CUDARL_GODOT_CPP_ROOT}")
if(godot_cpp_root STREQUAL "${repo}/external/godot-cpp" AND
   NOT EXISTS "${godot_cpp_root}/include/godot_cpp/godot.hpp")
    get_filename_component(godot_cpp_root "${repo}/../../external/godot-cpp" ABSOLUTE)
endif()
set(CUDARL_GODOT_CPP_LIBRARY "${godot_cpp_root}/bin/libgodot-cpp.windows.template_release.x86_64.lib"
    CACHE FILEPATH "Built Godot 4.4 Windows x64 C++ library")
if(NOT EXISTS "${CUDARL_GODOT_CPP_LIBRARY}")
    message(FATAL_ERROR "Missing Godot 4.4 C++ library at ${CUDARL_GODOT_CPP_LIBRARY}")
endif()

set(habitat_path "${repo}/flexible-tail-preview/habitat.json")
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
configure_file("../pond-flexible-tail-check/habitat_config.h.in" habitat_config.h @ONLY)

add_library(flexible_tail_live SHARED
    "${repo}/src/godot/flexible_tail_live.cpp"
    "${repo}/src/godot/register_flexible_tail_live.cpp"
    "${repo}/src/core/checkpoint.cpp")
target_compile_features(flexible_tail_live PRIVATE cxx_std_17)
target_compile_definitions(flexible_tail_live PRIVATE TYPED_METHOD_BIND WINDOWS_ENABLED CUDARL_RELIABLE_CHOOSER=1)
target_include_directories(flexible_tail_live PRIVATE "${repo}/src" "${repo}/src/nn" "${CMAKE_CURRENT_BINARY_DIR}")
target_include_directories(flexible_tail_live SYSTEM PRIVATE "${godot_cpp_root}/include"
    "${godot_cpp_root}/gen/include" "${godot_cpp_root}/gdextension")
target_link_libraries(flexible_tail_live PRIVATE "${CUDARL_GODOT_CPP_LIBRARY}")
if(MSVC)
    target_compile_options(flexible_tail_live PRIVATE /W4 /WX /permissive-)
endif()

set(godot_extension_directory "${repo}/flexible-tail-preview/addons/flexible_tail_live/bin")
add_custom_target(stage_flexible_tail_live
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${godot_extension_directory}"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "$<TARGET_FILE:flexible_tail_live>"
        "${godot_extension_directory}/$<TARGET_FILE_NAME:flexible_tail_live>"
    DEPENDS flexible_tail_live
    COMMENT "Staging the flexible-tail Godot extension"
    VERBATIM)

