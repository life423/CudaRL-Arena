#ifndef CUDARL_DEMOS_POND_PLAYBACK_H
#define CUDARL_DEMOS_POND_PLAYBACK_H
#include <memory>
#include <string>
#include "rl/pond_selector_policy.h"
namespace cudarl::demos {
struct PondFrame {deep_pond::State state;int selected=-1;};
class PondPlayback {
public:
    virtual ~PondPlayback()=default;
    virtual const PondFrame& frame() const=0;
    virtual const PondFrame& step()=0;
    virtual const char* backend() const=0;
    virtual const std::string& detail() const=0;
};
// preference: auto (CUDA health check, then CPU), cpu, or strict cuda.
// An absent optional module never prevents the portable CPU player from loading.
std::unique_ptr<PondPlayback> create_pond_playback(const rl::SelectorPolicy& policy,const std::string& preference="auto",
    const std::string& cuda_module={},uint32_t seed=7,uint32_t environment=0);
}
#endif
