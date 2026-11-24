#pragma once
#include "../pond-ppo/transition_trace.h"
#include <string>
#include <vector>
namespace pond_collection {
struct Inputs {std::vector<uint8_t> worlds,tasks;std::vector<float> residual,critic;std::string package;};
struct Samples {
    std::vector<float> obs,residual,mask,base,latent,logp,value,reward,done,truncated,final_value,last_value;
    std::vector<uint64_t> attempted,substeps;std::vector<int> reason;std::vector<Trace> trace;
};
struct Gap {double bound;uint32_t precise,free_steps,exact_steps,full_queries;};
struct Timing {double setup_ms=0,input_transfer_ms=0,physics_ms=0,inference_ms=0,collection_wall_ms=0,readback_ms=0;};
void collect_once(const Inputs&,Samples&,std::vector<Gap>&,Timing&,void(*issued)(int),void(*complete)(int));
std::string source();
}
