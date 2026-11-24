#pragma once
#include "../pond-clearance-dynamics/replay.h"
namespace swept {
struct Certificate {
    double bounds[10]{},start_lower=0,start_upper=0,minimum=0,bound=0;
    int measured[10]{},precise=0,exact_queries=0,free_steps=0,exact_steps=0,free_trials=0;
};
void load();
void launch(void*,const replay::Request*,void*,replay::Output*,int,cudaStream_t);
void attach(Certificate*);
size_t scratch();
replay::Output cpu(const void*,const float*,Certificate&);
}
