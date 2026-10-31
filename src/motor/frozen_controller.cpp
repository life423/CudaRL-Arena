#include "frozen_controller.h"
#include "frozen_contract.h"
#include "core/checkpoint.h"
#include <cmath>
#include <stdexcept>
namespace cudarl::motor {
namespace sm=straight_motor;namespace cp=checkpoint;namespace p=physics;
static_assert(movement::kVersion==6&&sm::kTaskVersion==5&&sm::kRewardVersion==2,"Frozen actor contract changed");
FrozenController::FrozenController():actor_({{42,32,32,2},{nn::Activation::Tanh,nn::Activation::Tanh,nn::Activation::Identity}}){}
bool FrozenController::load(const std::string& path){loaded_=false;paused_=true;error_.clear();
    try{const auto bytes=cp::read(path);cp::Reader r{bytes};
        if(r.value<uint32_t>()!=0x36434D46||r.value<uint32_t>()!=2||r.value<uint32_t>()!=6||r.value<uint32_t>()!=5||r.value<uint32_t>()!=2||
            r.value<uint32_t>()!=42||r.value<uint32_t>()!=2)throw std::runtime_error("Wrong frozen controller schema/model/task/shape");
        (void)r.value<uint32_t>(); // training update provenance; no optimizer loaded
        const float low=r.value<float>(),high=r.value<float>();
        if(!sm::supported_speed(low)||!sm::supported_speed(high)||low>high)throw std::runtime_error("Invalid validated speed range");
        (void)r.value<uint64_t>();const auto actor_checksum=r.value<uint64_t>();
        if(r.value<uint32_t>()!=kFrozenContractCount)throw std::runtime_error("Source contract count mismatch");
        for(const char* expected:kFrozenContracts){const auto hash=r.array<uint8_t>(64);if(std::string(hash.begin(),hash.end())!=expected)throw std::runtime_error("Frozen actor and compiled source contract differ");}
        auto params=r.array<float>(2498);r.finish();if(params.size()!=2498)throw std::runtime_error("Wrong parameter count");
        cp::Writer actor_bytes;for(float value:params){if(!std::isfinite(value))throw std::runtime_error("Nonfinite actor parameter");actor_bytes.value(value);}
        if(cp::checksum(actor_bytes.data)!=actor_checksum)throw std::runtime_error("Frozen actor checksum differs from selected checkpoint");
        actor_.params()=std::move(params);minimum_speed_=low;maximum_speed_=high;config_=sm::config_for_speed(high);
        loaded_=true;return reset(0);
    }catch(const std::exception& e){error_=e.what();return false;}
}
bool FrozenController::reset(int id){paused_=true;error_.clear();if(!loaded_){error_="Load the validated frozen actor first";return false;}if(id<0||id>3){error_="Case must be0..3";return false;}
    case_=id;world_=sm::initial_world(id);
    task_={};if(!sm::begin_forward(world_,config_,task_)){error_="Invalid evaluation reset";return false;}return true;
}
bool FrozenController::set_speed(float speed){
    if(!loaded_||!std::isfinite(speed)||speed<minimum_speed_||speed>maximum_speed_)return false;
    return sm::request_speed(world_,speed,config_,task_);
}
bool FrozenController::start(){if(!loaded_||finished()||!error_.empty())return false;paused_=false;return true;}
bool FrozenController::step(){if(!loaded_||paused_||finished()||!error_.empty())return false;
    float observation[42]{};if(!sm::observe(task_,world_,config_,observation)){error_="Invalid observation";paused_=true;return false;}
    actor_.forward(observation,1);sm::Action action;
    if(!sm::action_from_latent(actor_.output(),action)||!sm::step_state(world_,action.command)){error_="Physical/controller failure code "+std::to_string(world_.failure_code);paused_=true;return false;}
    const auto transition=sm::record_control(task_,world_,config_);if(!transition.accepted){error_="Task rejected native interval";paused_=true;return false;}
    if(transition.done)paused_=true;return true;
}
}
