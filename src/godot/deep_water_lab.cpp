#include "deep_water_lab.h"
#include <exception>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/classes/project_settings.hpp>
#include <godot_cpp/variant/vector3.hpp>
#include <godot_cpp/variant/quaternion.hpp>
#include <godot_cpp/variant/packed_vector3_array.hpp>
#include "cuda/device.h"
namespace godot {
void DeepWaterLab::_bind_methods() {
    ClassDB::bind_method(D_METHOD("start_training","environments","seed","checkpoint"),&DeepWaterLab::start_training,
        DEFVAL(1024),DEFVAL(7),DEFVAL("user://pond-policies/deep-v1/training.bin"));
    ClassDB::bind_method(D_METHOD("stop"),&DeepWaterLab::stop);
    ClassDB::bind_method(D_METHOD("set_paused","paused"),&DeepWaterLab::set_paused);
    ClassDB::bind_method(D_METHOD("set_updates_per_second","speed"),&DeepWaterLab::set_updates_per_second);
    ClassDB::bind_method(D_METHOD("save_training_now"),&DeepWaterLab::save_training_now);
    ClassDB::bind_method(D_METHOD("step"),&DeepWaterLab::step);
    ClassDB::bind_method(D_METHOD("get_state"),&DeepWaterLab::get_state);
    ClassDB::bind_method(D_METHOD("get_stats"),&DeepWaterLab::get_stats);
    ClassDB::bind_method(D_METHOD("get_evaluation_history","from"),&DeepWaterLab::get_evaluation_history,DEFVAL(0));
    ClassDB::bind_method(D_METHOD("get_error"),&DeepWaterLab::get_error);
}
bool DeepWaterLab::start_training(int environments,int seed,String checkpoint) {
    stop();error_=String();
    if(environments<=0 || environments>32768) {error_="Invalid deep-water population";return false;}
    const auto device=cudarl::probe_device();
    if(!device.available) {error_=String("CUDA unavailable: ")+device.error.c_str();return false;}
    try {
        cudarl::rl::SwimmerConfig cfg;cfg.num_envs=environments;cfg.seed=static_cast<uint32_t>(seed);
        cfg.total_updates=0;cfg.evaluation_every=25;
        cfg.training_checkpoint_path=ProjectSettings::get_singleton()->globalize_path(checkpoint).utf8().get_data();
        actor_=std::make_unique<cudarl::nn::CpuMlp<float>>(cudarl::rl::swimmer_specification(cudarl::deep::kAct,cudarl::deep::kObs));
        actor_->params()=cudarl::rl::DeepWaterTrainer::initial_actor_params(cfg.seed);
        state_=cudarl::deep::initial(cfg.seed,0);
        session_=std::make_unique<cudarl::demos::TrainingSession<cudarl::rl::DeepWaterTrainer>>(cfg,0,true);
        return true;
    } catch(const std::exception& e) {stop();error_=e.what();return false;}
}
void DeepWaterLab::stop() {
    if(session_) session_->stop();session_.reset();actor_.reset();history_.clear();evaluations_.clear();
    policy_version_=0;best_version_=0;policy_update_=0;training_ms_=0;best_={};
}
void DeepWaterLab::set_paused(bool paused) {if(session_) session_->set_paused(paused);}
void DeepWaterLab::set_updates_per_second(double speed) {if(session_) session_->set_updates_per_second(speed);}
void DeepWaterLab::save_training_now() {if(session_) session_->request_save();}
String DeepWaterLab::get_error() const {return !error_.is_empty()?error_:(session_?String(session_->error().c_str()):String());}
void DeepWaterLab::collect() {
    if(!session_) return;const auto first=history_.size();session_->history_since(first,history_);
    for(size_t i=first;i<history_.size();++i) {
        const auto& s=history_[i];training_ms_+=s.update_ms;
        if(s.evaluation_count>0 && (evaluations_.empty() || s.evaluation_update>evaluations_.back().evaluation_update)) evaluations_.push_back(s);
    }
    session_->latest_best_policy(best_version_,best_);
}
Dictionary DeepWaterLab::step() {
    if(actor_) {
        std::vector<float> parameters;
        if(session_ && session_->latest_policy(policy_version_,parameters,&policy_update_)) actor_->params()=std::move(parameters);
        float obs[cudarl::deep::kObs];cudarl::deep::observe(state_,obs);actor_->forward(obs,1);
        cudarl::deep::advance(state_,actor_->output());
    }
    return get_state();
}
Dictionary DeepWaterLab::get_state() const {
    Dictionary d;d["ready"]=static_cast<bool>(actor_);if(!actor_) return d;
    const auto& p=state_.body.position;const auto& q=state_.body.orientation;
    d["position"]=Vector3(p.x,p.y,p.z);d["orientation"]=Quaternion(q.x,q.y,q.z,q.w);
    d["food"]=Vector3(state_.food.x,state_.food.y,state_.food.z);
    d["eaten"]=state_.eaten;d["elapsed"]=state_.steps*cudarl::deep::dt;
    d["seconds_since_food"]=state_.since_food*cudarl::deep::dt;
    d["depth_cm"]=-p.y*4.;d["food_depth_cm"]=-state_.food.y*4.;
    d["speed_cm_s"]=100.*cudarl::deep::norm({state_.body.velocity[0],state_.body.velocity[1],state_.body.velocity[2]});
    const auto shape=cudarl::deep::shape(state_.body);Array links;
    for(const auto& link:shape.link) {
        Dictionary item;const auto& r=link.q;const auto c=link.center/cudarl::deep::scale;
        item["position"]=Vector3(c.x,c.y,c.z);item["orientation"]=Quaternion(r.x,r.y,r.z,r.w);links.push_back(item);
    }
    d["links"]=links;return d;
}
Dictionary DeepWaterLab::get_stats() {
    collect();Dictionary d;
    const auto save=session_?session_->checkpoint_status():cudarl::checkpoint::Status{};
    d["training_ready"]=session_ && session_->ready();d["paused"]=session_ && session_->paused();
    d["resumed"]=save.resumed;d["loaded_update"]=save.loaded_update;d["saved_update"]=save.saved_update;
    d["save_error"]=String(save.error.c_str());d["policy_update"]=policy_update_;
    d["update"]=history_.empty()?0:history_.back().update;
    d["env_steps"]=history_.empty()?int64_t(0):static_cast<int64_t>(history_.back().env_steps);
    d["update_ms"]=history_.empty()?0.:history_.back().update_ms;d["training_seconds"]=training_ms_*.001;
    d["background_swimmers"]=session_?session_->config().num_envs:0;
    d["current_food_per_minute"]=evaluations_.empty()?0.:evaluations_.back().evaluation_foods*60./(64*evaluations_.back().evaluation_seconds);
    d["best_food_per_minute"]=best_.food_per_minute();d["best_update"]=best_.update;
    d["evaluation_count"]=static_cast<int64_t>(evaluations_.size());
    d["checkpoint_path"]=session_?String(session_->config().training_checkpoint_path.c_str()):String();return d;
}
Array DeepWaterLab::get_evaluation_history(int from) {
    collect();Array a;for(size_t i=static_cast<size_t>(std::max(0,from));i<evaluations_.size();++i) {
        const auto& s=evaluations_[i];Dictionary d;d["update"]=s.evaluation_update;
        d["food_per_minute"]=s.evaluation_foods*60./(s.evaluation_count*s.evaluation_seconds);a.push_back(d);
    }
    return a;
}
}
