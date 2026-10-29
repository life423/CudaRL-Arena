#ifndef CUDARL_GODOT_DEEP_POND_LAB_H
#define CUDARL_GODOT_DEEP_POND_LAB_H
#include <memory>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/string.hpp>
#include "demos/pond_playback.h"
#include "nn/mlp_cpu.h"
#include "demos/feeding_session.h"
#if CUDARL_WITH_CUDA
#include "demos/ppo_session.h"
#include "rl/swimmer_ppo.h"
#endif
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
    bool watch_controller(String path,String backend="auto");
    bool watch_progress_controller(String path,String backend="auto");
    bool set_untrained(bool enabled);
    bool set_gentle_current(bool enabled,bool drifting_food=false);
    bool set_steady_current(double speed_cm_s,bool drifting_food=false);
    bool start_feeding_learning(String policy,String checkpoint,int environments=1024,String backend="auto");
    bool start_arch_learning(String source_checkpoint,String checkpoint,int environments=1024,String backend="auto");
    Dictionary get_arch_geometry() const;
    int drop_food(Vector3 position);
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
#if CUDARL_WITH_CUDA
    std::unique_ptr<cudarl::demos::TrainingSession<cudarl::rl::DeepPondTrainer>> session_;
#endif
    std::unique_ptr<cudarl::demos::FeedingSession> feeding_;
    std::unique_ptr<cudarl::nn::CpuMlp<float>> actor_;
    std::unique_ptr<cudarl::demos::PondPlayback> playback_;
    cudarl::rl::SelectorPolicy controller_;
    cudarl::rl::ProgressPolicy progress_controller_;
    bool use_progress_controller_=false;
    String controller_path_;
    String controller_backend_="auto";
    bool showcase_untrained_=false;
    int selected_food_=-1;
    cudarl::deep_pond::State state_;
    int policy_update_=0;
#if CUDARL_WITH_CUDA
    uint64_t policy_version_=0,best_version_=0;
    int applied_best_update_=-1;
    bool watching_best_=true;
    double training_ms_=0;
    std::vector<cudarl::rl::SwimmerStats> history_,evaluations_;
    cudarl::rl::EvaluatedPolicy best_;
#endif
    String error_;
    void collect();
};
}
#endif
