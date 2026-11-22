#ifndef CUDARL_FROZEN_CONTROLLER_H
#define CUDARL_FROZEN_CONTROLLER_H
#include "motor/straight_task.h"
#include "nn/mlp_cpu.h"
#include <string>
#include <string_view>
#include <vector>
namespace cudarl::motor {
// One physical state, one frozen actor; no optimizer or background worker.
class FrozenController {
    straight_motor::World world_{};
    straight_motor::ForwardTask task_{};
    straight_motor::ForwardConfig config_{};
    nn::CpuMlp<float> actor_;
    bool loaded_=false,paused_=true,continuous_=false;
    uint64_t viewing_controls_=0,viewing_substeps_=0,loops_=0;
    int case_=0;
    contact::Arena arena_{};
    float minimum_speed_=.008f,maximum_speed_=.008f;
    std::string error_;
    uint32_t training_update_=0;
    uint64_t source_protocol_checksum_=0;
public:
    FrozenController();
    bool load(const std::string& path);
    static std::vector<std::string_view> source_contract_hashes();
    uint32_t observation_schema() const { return config_.observation_schema; }
    uint32_t training_update() const { return training_update_; }
    uint64_t source_protocol_checksum() const { return source_protocol_checksum_; }
    static constexpr int substeps_per_control() { return joint_motor::kSubstepsPerControl; }
    int benchmark_controls() const { return config_.maximum_controls; }
    bool reset(int case_id=0);
    bool start();
    bool set_speed(float speed);
    bool configure_arena(contact::Arena arena){if(!loaded_||!paused_||viewing_controls_!=0||!contact::valid_arena(arena))return false;arena_=arena;world_.arena=arena;return true;}
    contact::Arena arena()const{return arena_;}
    uint32_t action_mapping()const{return joint_motor::kActionVersion;}
    void set_continuous(bool enabled){pause();continuous_=enabled;}
    bool continuous()const{return continuous_;}
    uint64_t viewing_controls()const{return viewing_controls_;}
    uint64_t viewing_substeps()const{return viewing_substeps_;}
    uint64_t loops()const{return loops_;}
    bool benchmark_complete()const{return task_.outcome!=straight_motor::Outcome::Running;}
    float target_speed()const{return config_.target_speed_m_s;}
    float minimum_speed()const{return minimum_speed_;}
    float maximum_speed()const{return maximum_speed_;}
    const std::vector<float>& actor_parameters()const{return actor_.params();}
    void pause(){paused_=true;}
    bool step(); // exactly one50ms control; refuses paused/unloaded/finished
    bool loaded()const{return loaded_;}
    bool paused()const{return paused_;}
    bool finished()const{return !continuous_&&benchmark_complete();}
    const std::string& error()const{return error_;}
    const straight_motor::World& world()const{return world_;}
    const straight_motor::ForwardTask& task()const{return task_;}
    straight_motor::Metrics metrics()const{return straight_motor::measure(task_,world_);}
    int case_id()const{return case_;}
};
}
#endif
