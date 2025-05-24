cmake_minimum_required(VERSION 3.24)

# ──────────────── Catch2 for testing ────────────────
include(FetchContent)
FetchContent_Declare(
  catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.4.0
)
FetchContent_MakeAvailable(catch2)

# ──────────────── C++ Tests ────────────────
add_executable(test_environment test_environment.cpp)
target_link_libraries(test_environment PRIVATE 
  cudarl_core
  Catch2::Catch2WithMain
)

# Register tests with CTest
include(CTest)
include(Catch)
catch_discover_tests(test_environment)

# ──────────────── Python Tests ────────────────
add_custom_target(python_tests
  COMMAND ${Python3_EXECUTABLE} -m unittest discover -s ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS cudarl_python_bindings
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMENT "Running Python tests"
)

# ──────────────── Run All Tests ────────────────
add_custom_target(run_all_tests
  DEPENDS test_environment python_tests
  COMMENT "Running all tests"
)