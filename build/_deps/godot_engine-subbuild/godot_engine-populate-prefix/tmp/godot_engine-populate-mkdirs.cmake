# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "C:/Users/aiand/Documents/programming/CudaRL-Arena/build/_deps/godot_engine-src")
  file(MAKE_DIRECTORY "C:/Users/aiand/Documents/programming/CudaRL-Arena/build/_deps/godot_engine-src")
endif()
file(MAKE_DIRECTORY
  "C:/Users/aiand/Documents/programming/CudaRL-Arena/build/_deps/godot_engine-build"
  "C:/Users/aiand/Documents/programming/CudaRL-Arena/build/_deps/godot_engine-subbuild/godot_engine-populate-prefix"
  "C:/Users/aiand/Documents/programming/CudaRL-Arena/build/_deps/godot_engine-subbuild/godot_engine-populate-prefix/tmp"
  "C:/Users/aiand/Documents/programming/CudaRL-Arena/build/_deps/godot_engine-subbuild/godot_engine-populate-prefix/src/godot_engine-populate-stamp"
  "C:/Users/aiand/Documents/programming/CudaRL-Arena/build/_deps/godot_engine-subbuild/godot_engine-populate-prefix/src"
  "C:/Users/aiand/Documents/programming/CudaRL-Arena/build/_deps/godot_engine-subbuild/godot_engine-populate-prefix/src/godot_engine-populate-stamp"
)

set(configSubDirs Debug)
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "C:/Users/aiand/Documents/programming/CudaRL-Arena/build/_deps/godot_engine-subbuild/godot_engine-populate-prefix/src/godot_engine-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "C:/Users/aiand/Documents/programming/CudaRL-Arena/build/_deps/godot_engine-subbuild/godot_engine-populate-prefix/src/godot_engine-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
