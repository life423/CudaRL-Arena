#ifndef CUDARL_GODOT_DEEP_POND_LAB_H
#define CUDARL_GODOT_DEEP_POND_LAB_H
#include <memory>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/string.hpp>
#include "demos/ppo_session.h"
#include "rl/swimmer_ppo.h"
#include "nn/mlp_cpu.h"
namespace godot {
class DeepPondLab:public RefCounted {
    GDCLASS(DeepPondLab,RefCounted);
protected:
    static void _bind_methods();
public:
    ~DeepPondLab() override {stop();}
    bool start_training(int environments=1024,int seed=7,String checkpoint="user://pond-policies/deep-pond-v1/training.bin",
        String source="user://pond-policies/deep-v1/training.bin");
    void stop();
    void set_paused(bool paused);
    void set_updates_per_second(double speed);
    void save_training_now();
    void set_watch_best(bool enabled);
    Dictionary step();
    Dictionary get_state() const;
    Dictionary get_stats();
    Array get_evaluation_history(int from=0);
    String get_error() const;
private:
    std::unique_ptr<cudarl::demos::TrainingSession<cudarl::rl::DeepPondTrainer>> session_;
    std::unique_ptr<cudarl::nn::CpuMlp<float>> actor_;
    cudarl::deep_pond::State state_;
    uint64_t policy_version_=0,best_version_=0;
    int policy_update_=0;
    int applied_best_update_=-1;
    bool watching_best_=true;
    double training_ms_=0;
    String error_;
    std::vector<cudarl::rl::SwimmerStats> history_,evaluations_;
    cudarl::rl::EvaluatedPolicy best_;
    void collect();
};
}
#endif
