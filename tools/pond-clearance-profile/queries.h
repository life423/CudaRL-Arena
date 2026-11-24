#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace query_profile {
constexpr int worlds=512,world_bytes=1384,repeats=4;
constexpr double tolerance=.00001;
struct Link { double low,high,point[3],gradient[2]; int certified; };
struct Summary {
    double low,high,max_y;
    uint32_t visits;
    int valid,certified,failure;
    Link link[3];
};
using Launch=void(*)(const void*,Summary*,int,cudaStream_t);
}
namespace baseline {
void launch(const void*,query_profile::Summary*,int,cudaStream_t);
query_profile::Summary cpu(const void*);
double make_fixtures(void* dest);
}
namespace candidate {void launch(const void*,query_profile::Summary*,int,cudaStream_t);}
