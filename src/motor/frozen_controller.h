#ifndef CUDARL_FROZEN_CONTROLLER_H
#define CUDARL_FROZEN_CONTROLLER_H
#include "motor/straight_task.h"
#include "nn/mlp_cpu.h"
#include <string>
namespace cudarl::motor {
// One physical state, one frozen actor; no optimizer or background worker.
class FrozenController {
    straight_motor::World world_{};
    straight_motor::ForwardTask task_{};
    straight_motor::ForwardConfig config_{};
    nn::CpuMlp<float> actor_;
    bool loaded_=false,paused_=true;
    int case_=0;
    float minimum_speed_=.008f,maximum_speed_=.008f;
    std::string error_;
public:
    FrozenController();
    bool load(const std::string& path);
    bool reset(int case_id=0);
    bool start();
    bool set_speed(float speed);
    float target_speed()const{return config_.target_speed_m_s;}
    float minimum_speed()const{return minimum_speed_;}
    float maximum_speed()const{return maximum_speed_;}
    const std::vector<float>& actor_parameters()const{return actor_.params();}
    void pause(){paused_=true;}
    bool step(); // exactly one50ms control; refuses paused/unloaded/finished
    bool loaded()const{return loaded_;}
    bool paused()const{return paused_;}
    bool finished()const{return task_.outcome!=straight_motor::Outcome::Running;}
    const std::string& error()const{return error_;}
    const straight_motor::World& world()const{return world_;}
    const straight_motor::ForwardTask& task()const{return task_;}
    straight_motor::Metrics metrics()const{return straight_motor::measure(task_,world_);}
    int case_id()const{return case_;}
};
}
#endif
