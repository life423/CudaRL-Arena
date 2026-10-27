#ifndef CUDARL_GODOT_TADPOLE_LAB_H
#define CUDARL_GODOT_TADPOLE_LAB_H
#include <memory>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/string.hpp>
#include "demos/tadpole_pond.h"
#include "demos/foraging_pond.h"
#include "demos/ppo_session.h"
#include "rl/swimmer_ppo.h"
namespace godot {
class TadpoleLab:public RefCounted {
    GDCLASS(TadpoleLab,RefCounted);
protected:
    static void _bind_methods();
public:
    ~TadpoleLab() override {stop();}
    bool watch_policy(const PackedFloat32Array& weights,int seed=1);
    bool watch_foraging_checkpoint(String path,int seed=1);
    bool start_training(int num_envs=1024,int seed=1,double updates_per_second=10.0,
                        String checkpoint_path="user://pond-policies/regional-v1/best.json",bool shuffle_inputs=false,bool resume_training=true);
    void save_training_now();
    bool set_watch_best(bool enabled);
    void set_paused(bool paused);
    void set_updates_per_second(double updates_per_second);
    void stop();
    Dictionary get_stats();
    Array get_evaluation_history(int from=0);
    Dictionary step();
    Dictionary get_state() const;
    void reset(int seed=1);
    void move_food();
    String get_error() const;
private:
    std::unique_ptr<cudarl::demos::TadpolePond> pond_;
    std::unique_ptr<cudarl::demos::ForagingPond> foraging_;
    String error_;
    std::unique_ptr<cudarl::demos::TrainingSession<cudarl::rl::ForagingTrainer>> session_;
    std::vector<cudarl::rl::SwimmerStats> history_;
    uint64_t policy_version_=0;
    int applied_update_=0;
    double training_ms_=0;
    std::vector<cudarl::rl::SwimmerStats> evaluations_;
    void collect_stats();
    cudarl::rl::EvaluatedPolicy best_;
    String checkpoint_path_,checkpoint_error_;
    bool best_saved_=false,checkpoint_blocked_=false,watching_best_=false;
    double starting_best_score_=0;
    uint64_t session_best_version_=0,best_revision_=0,applied_best_revision_=0;
    void load_best();
    void save_best();
};
}
#endif
