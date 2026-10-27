#ifndef CUDARL_DEMOS_POND_TARGET_H
#define CUDARL_DEMOS_POND_TARGET_H
#include "envs/swimmer/deep_pond.h"
namespace cudarl::demos {
// A deliberately programmed evaluation baseline, not a learned selector.
CUDARL_HD inline int nearest_pond_food(const float* observation) {
    int best=0;float distance=1e30f;
    for(int f=0;f<3;++f) {
        const auto* p=observation+3*f;const float d=p[0]*p[0]+p[1]*p[1]+p[2]*p[2];
        if(d<distance) {best=f;distance=d;}
    }
    return best;
}
// Promote the selected *physical* slot; remaining slots retain canonical order.
// Only input coordinates change: world food, body state and torque outputs do not.
CUDARL_HD inline void primary_pond_food(float* observation,int selected) {
    float old[9];for(int i=0;i<9;++i) old[i]=observation[i];
    for(int j=0;j<3;++j) observation[j]=old[selected*3+j];
    int next=1;
    for(int f=0;f<3;++f) if(f!=selected) {
        for(int j=0;j<3;++j) observation[next*3+j]=old[f*3+j];++next;
    }
}
}
#endif
