#pragma once
#include <cuda_runtime.h>
#include <cstdint>
namespace replay {
constexpr int worlds=512,world_bytes=1384,task_bytes=368,ticks=10;
struct Request {float value[4];int holding,direct;};
struct Step {
    float body[3][23],held[4],rate[4],actual[4],applied[4],requested[4];
    double energy[16],low,high,force[3][3];int mode[3];
    uint64_t attempts,accepted;
    int valid,failed,iterations;
};
struct Output {Step steps[ticks];float command[4];uint64_t controls,substeps;int success;};
using Issue=void(*)(void*,const Request*,void*,Output*,int,cudaStream_t);
}
namespace baseline {void load();void launch(void*,const replay::Request*,void*,replay::Output*,int,cudaStream_t);size_t scratch();}
namespace candidate {void load();void launch(void*,const replay::Request*,void*,replay::Output*,int,cudaStream_t);size_t scratch();replay::Output cpu(const void*,const float*);int holding(const void*);}
