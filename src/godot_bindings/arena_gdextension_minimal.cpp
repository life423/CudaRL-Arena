#include "arena_gdextension_minimal.h"
#include <godot_cpp/variant/utility_functions.hpp>
#include <cuda_runtime.h>

using namespace godot;

void ArenaGDExtension::_bind_methods() {
    ClassDB::bind_method(D_METHOD("hello_cuda"), &ArenaGDExtension::hello_cuda);
    ClassDB::bind_method(D_METHOD("run_training", "episodes"), &ArenaGDExtension::run_training);
}

ArenaGDExtension::ArenaGDExtension() {
    // Constructor implementation
}

ArenaGDExtension::~ArenaGDExtension() {
    // Destructor implementation
}

String ArenaGDExtension::hello_cuda() {
    int n = 0;
    cudaGetDeviceCount(&n);
    return String("CUDA devices: ") + String::num_int64(n);
}

void ArenaGDExtension::run_training(int episodes) {
    UtilityFunctions::print("Training for ", episodes, " episodes (stub)");
}