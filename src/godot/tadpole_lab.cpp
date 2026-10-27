#include "tadpole_lab.h"
#include <exception>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/packed_vector2_array.hpp>
#include "cuda/device.h"
namespace godot {
void TadpoleLab::_bind_methods() {
    ClassDB::bind_method(D_METHOD("watch_policy","weights","seed"),&TadpoleLab::watch_policy,DEFVAL(1));
    ClassDB::bind_method(D_METHOD("step"),&TadpoleLab::step);
    ClassDB::bind_method(D_METHOD("get_state"),&TadpoleLab::get_state);
    ClassDB::bind_method(D_METHOD("reset","seed"),&TadpoleLab::reset,DEFVAL(1));
    ClassDB::bind_method(D_METHOD("move_food"),&TadpoleLab::move_food);
    ClassDB::bind_method(D_METHOD("get_error"),&TadpoleLab::get_error);
    ClassDB::bind_method(D_METHOD("start_training","num_envs","seed","updates_per_second"),&TadpoleLab::start_training,DEFVAL(1024),DEFVAL(1),DEFVAL(10.0));
    ClassDB::bind_method(D_METHOD("set_paused","paused"),&TadpoleLab::set_paused);
    ClassDB::bind_method(D_METHOD("set_updates_per_second","updates_per_second"),&TadpoleLab::set_updates_per_second);
    ClassDB::bind_method(D_METHOD("stop"),&TadpoleLab::stop);
    ClassDB::bind_method(D_METHOD("get_stats"),&TadpoleLab::get_stats);
    ClassDB::bind_method(D_METHOD("get_evaluation_history","from"),&TadpoleLab::get_evaluation_history,DEFVAL(0));
}
bool TadpoleLab::watch_policy(const PackedFloat32Array& weights,int seed) {
    stop();error_=String();
    try {
        auto pond=std::make_unique<cudarl::demos::TadpolePond>();
        std::vector<float> parameters;
        if(!weights.is_empty()) parameters.assign(weights.ptr(),weights.ptr()+weights.size());
        pond->set_policy(parameters);pond->reset(static_cast<uint32_t>(seed));pond_=std::move(pond);
        return true;
    } catch(const std::exception& e) {error_=e.what();return false;}
}
bool TadpoleLab::start_training(int num_envs,int seed,double updates_per_second) {
    stop();error_=String();
    if(num_envs<=0 || num_envs>32768) {error_="Invalid background swimmer count";return false;}
    const auto device=cudarl::probe_device();
    if(!device.available) {error_=String("CUDA unavailable: ")+device.error.c_str();return false;}
    try {
        cudarl::rl::SwimmerConfig cfg;cfg.num_envs=num_envs;cfg.seed=static_cast<uint32_t>(seed);
        cfg.total_updates=0; // Keep learning until the viewer explicitly stops the session.
        cfg.evaluation_every=25;
        cfg.body=cudarl::swimmer::body_preset(8);
        foraging_=std::make_unique<cudarl::demos::ForagingPond>(cfg.body);
        foraging_->set_policy(cudarl::rl::ForagingTrainer::initial_actor_params(cfg.seed));
        foraging_->reset(cfg.seed);
        session_=std::make_unique<cudarl::demos::TrainingSession<cudarl::rl::ForagingTrainer>>(cfg,updates_per_second,true);
        return true;
    } catch(const std::exception& e) {stop();error_=e.what();return false;}
}
void TadpoleLab::set_paused(bool paused) {if(session_) session_->set_paused(paused);}
void TadpoleLab::set_updates_per_second(double updates_per_second) {if(session_) session_->set_updates_per_second(updates_per_second);}
void TadpoleLab::stop() {
    session_.reset();pond_.reset();foraging_.reset();history_.clear();policy_version_=0;applied_update_=0;
    evaluations_.clear();training_ms_=0;
}
String TadpoleLab::get_error() const {
    if(!error_.is_empty()) return error_;
    return session_?String(session_->error().c_str()):String();
}
void TadpoleLab::collect_stats() {
    const size_t previous=history_.size();
    if(session_) session_->history_since(previous,history_);
    for(size_t i=previous;i<history_.size();++i) {
        const auto& stats=history_[i];
        training_ms_+=stats.update_ms;
        if(stats.evaluation_count>0 && (evaluations_.empty() || stats.evaluation_update>evaluations_.back().evaluation_update))
            evaluations_.push_back(stats);
    }
}
Array TadpoleLab::get_evaluation_history(int from) {
    collect_stats();
    Array result;
    for(size_t i=static_cast<size_t>(std::max(0,from));i<evaluations_.size();++i) {
        const auto& stats=evaluations_[i];
        Dictionary point;
        point["update"]=stats.evaluation_update;
        point["successes"]=stats.evaluation_successes;
        point["count"]=stats.evaluation_count;
        point["seconds"]=stats.evaluation_seconds;
        point["foods"]=stats.evaluation_foods;
        point["food_per_minute"]=stats.evaluation_foods*60.0/(stats.evaluation_count*stats.evaluation_seconds);
        result.push_back(point);
    }
    return result;
}
Dictionary TadpoleLab::get_stats() {
    collect_stats();
    Dictionary d;
    d["training"]=static_cast<bool>(session_);
    d["background_swimmers"]=session_?session_->config().num_envs:0;
    d["update"]=history_.empty()?0:history_.back().update;
    d["policy_update"]=applied_update_;
    d["env_steps"]=history_.empty()?int64_t(0):static_cast<int64_t>(history_.back().env_steps);
    d["paused"]=session_ && session_->paused();
    d["finished"]=session_ && session_->finished();
    d["training_seconds"]=training_ms_*0.001;
    d["update_ms"]=history_.empty()?0.0:history_.back().update_ms;
    d["food_count"]=foraging_?3:(pond_?1:0);
    d["policy_inputs"]=foraging_?cudarl::foraging::kObs:cudarl::swimmer::kObs;
    return d;
}
Dictionary TadpoleLab::get_state() const {
    Dictionary d;d["ready"]=static_cast<bool>(pond_) || static_cast<bool>(foraging_);
    if(!pond_ && !foraging_) return d;
    const auto& s=foraging_?foraging_->state().body:pond_->state();
    d["x"]=s.x;d["y"]=s.y;d["heading"]=s.heading;
    d["joint0"]=s.angle[0];d["joint1"]=s.angle[1];
    PackedVector2Array foods;
    if(foraging_) {
        const auto& state=foraging_->state();
        for(int i=0;i<cudarl::foraging::kFoods;++i) foods.push_back(Vector2(state.food_x[i],state.food_y[i]));
        d["eaten"]=state.eaten;d["elapsed"]=foraging_->elapsed();
        d["seconds_since_food"]=state.since_food_steps*cudarl::swimmer::kDt;
        d["eaten_slot"]=state.last_eaten;
    } else {
        foods.push_back(Vector2(s.food_x,s.food_y));
        d["eaten"]=pond_->eaten();d["elapsed"]=pond_->elapsed();
        d["seconds_since_food"]=s.steps*cudarl::swimmer::kDt;d["eaten_slot"]=-1;
    }
    d["foods"]=foods;
    d["food_x"]=foods[0].x;d["food_y"]=foods[0].y; // Archived one-food viewer compatibility.
    // Inertial model scale is 0.04 metres per model unit.
    d["speed_cm_s"]=4.0*std::sqrt(s.velocity[0]*s.velocity[0]+s.velocity[1]*s.velocity[1]);
    return d;
}
Dictionary TadpoleLab::step() {
    bool ate=false;
    try {
        if(session_ && foraging_) {
            std::vector<float> parameters;
            if(session_->latest_policy(policy_version_,parameters)) {
                foraging_->set_policy(parameters);
                applied_update_=static_cast<int>(policy_version_-1);
            }
        }
        if(foraging_) ate=foraging_->step();
        else if(pond_) ate=pond_->step();
    }
    catch(const std::exception& e) {error_=e.what();}
    Dictionary d=get_state();d["ate"]=ate;return d;
}
void TadpoleLab::reset(int seed) {
    if(pond_) pond_->reset(static_cast<uint32_t>(seed));
    if(foraging_) foraging_->reset(static_cast<uint32_t>(seed));
}
void TadpoleLab::move_food() {if(pond_) pond_->place_food();}
}
