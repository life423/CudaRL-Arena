#include "deep_pond_lab.h"
#include <exception>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/classes/project_settings.hpp>
#include <godot_cpp/variant/vector3.hpp>
#include <godot_cpp/variant/quaternion.hpp>
#include <godot_cpp/variant/packed_vector3_array.hpp>
#include "cuda/device.h"
namespace godot {
void DeepPondLab::_bind_methods() {
    ClassDB::bind_method(D_METHOD("start_training","environments","seed","checkpoint","source"),&DeepPondLab::start_training,
        DEFVAL(1024),DEFVAL(7),DEFVAL("user://pond-policies/deep-pond-v1/training.bin"),DEFVAL("user://pond-policies/deep-v1/training.bin"));
    ClassDB::bind_method(D_METHOD("stop"),&DeepPondLab::stop);
    ClassDB::bind_method(D_METHOD("watch_controller","path"),&DeepPondLab::watch_controller);
    ClassDB::bind_method(D_METHOD("set_paused","paused"),&DeepPondLab::set_paused);
    ClassDB::bind_method(D_METHOD("set_updates_per_second","speed"),&DeepPondLab::set_updates_per_second);
    ClassDB::bind_method(D_METHOD("save_training_now"),&DeepPondLab::save_training_now);
    ClassDB::bind_method(D_METHOD("set_watch_best","enabled"),&DeepPondLab::set_watch_best);
    ClassDB::bind_method(D_METHOD("step"),&DeepPondLab::step);
    ClassDB::bind_method(D_METHOD("get_state"),&DeepPondLab::get_state);
    ClassDB::bind_method(D_METHOD("get_stats"),&DeepPondLab::get_stats);
    ClassDB::bind_method(D_METHOD("get_evaluation_history","from"),&DeepPondLab::get_evaluation_history,DEFVAL(0));
    ClassDB::bind_method(D_METHOD("get_error"),&DeepPondLab::get_error);
}
bool DeepPondLab::start_training(int environments,int seed,String checkpoint,String source) {
    stop();error_=String();
    if(environments<=0 || environments>32768) {error_="Invalid deep-water population";return false;}
    const auto device=cudarl::probe_device();
    if(!device.available) {error_=String("CUDA unavailable: ")+device.error.c_str();return false;}
    try {
        cudarl::rl::SwimmerConfig cfg;cfg.num_envs=environments;cfg.seed=static_cast<uint32_t>(seed);
        cfg.total_updates=0;cfg.evaluation_every=25;
        cfg.training_checkpoint_path=ProjectSettings::get_singleton()->globalize_path(checkpoint).utf8().get_data();
        cfg.deep_source_checkpoint=source.is_empty()?std::string():std::string(ProjectSettings::get_singleton()->globalize_path(source).utf8().get_data());
        // Retain a stable food ordering while adapting the inherited single-goal
        // motor controller. Per-tick permutations make its destination jump.
        cfg.shuffle_food_inputs=false;
        actor_=std::make_unique<cudarl::nn::CpuMlp<float>>(cudarl::rl::swimmer_specification(cudarl::deep::kAct,cudarl::deep_pond::kObs));
        actor_->params()=cudarl::rl::DeepPondTrainer::initial_actor_params(cfg.seed);
        state_=cudarl::deep_pond::initial(cfg.seed,0);
        session_=std::make_unique<cudarl::demos::TrainingSession<cudarl::rl::DeepPondTrainer>>(cfg,0,true);
        return true;
    } catch(const std::exception& e) {stop();error_=e.what();return false;}
}
void DeepPondLab::stop() {
    if(session_) session_->stop();session_.reset();actor_.reset();history_.clear();evaluations_.clear();
    selector_actor_.reset();controller_={};controller_path_=String();selected_food_=-1;
    policy_version_=0;best_version_=0;policy_update_=0;training_ms_=0;best_={};applied_best_update_=-1;watching_best_=true;
}
bool DeepPondLab::watch_controller(String path) {
    stop();error_=String();
    try {
        controller_=cudarl::rl::SelectorPolicy::load(ProjectSettings::get_singleton()->globalize_path(path).utf8().get_data());
        actor_=std::make_unique<cudarl::nn::CpuMlp<float>>(cudarl::rl::swimmer_specification(4,33));actor_->params()=controller_.motor;
        selector_actor_=std::make_unique<cudarl::nn::CpuMlp<float>>(cudarl::rl::selector_spec());selector_actor_->params()=controller_.selector;
        state_=cudarl::deep_pond::initial(7,0);policy_update_=controller_.motor_update;controller_path_=path;return true;
    } catch(const std::exception& e) {stop();error_=e.what();return false;}
}
void DeepPondLab::set_paused(bool paused) {if(session_) session_->set_paused(paused);}
void DeepPondLab::set_updates_per_second(double speed) {if(session_) session_->set_updates_per_second(speed);}
void DeepPondLab::save_training_now() {if(session_) session_->request_save();}
void DeepPondLab::set_watch_best(bool enabled) {watching_best_=enabled;policy_version_=0;applied_best_update_=-1;}
String DeepPondLab::get_error() const {return !error_.is_empty()?error_:(session_?String(session_->error().c_str()):String());}
void DeepPondLab::collect() {
    if(!session_) return;const auto first=history_.size();session_->history_since(first,history_);
    for(size_t i=first;i<history_.size();++i) {
        const auto& s=history_[i];training_ms_+=s.update_ms;
        if(s.evaluation_count>0 && (evaluations_.empty() || s.evaluation_update>evaluations_.back().evaluation_update)) evaluations_.push_back(s);
    }
    session_->latest_best_policy(best_version_,best_);
}
Dictionary DeepPondLab::step() {
    if(actor_) {
        std::vector<float> parameters;
        collect();
        if(selector_actor_) {
            if(state_.steps%controller_.decision_steps==0) {
                float observation[36],scores[3];cudarl::pond_selector::observe(state_,selected_food_,observation);
                for(int f=0;f<3;++f) {
                    float features[32];cudarl::pond_selector::features(observation,f,features);selector_actor_->forward(features,1);scores[f]=selector_actor_->output()[0];
                }
                selected_food_=0;for(int f=1;f<3;++f) if(scores[f]>scores[selected_food_]) selected_food_=f;
            }
        } else if(watching_best_ && best_.valid()) {
            if(applied_best_update_!=best_.update) {actor_->params()=best_.parameters;policy_update_=best_.update;applied_best_update_=best_.update;}
        } else if(session_ && session_->latest_policy(policy_version_,parameters,&policy_update_)) actor_->params()=std::move(parameters);
        float obs[cudarl::deep_pond::kObs];cudarl::deep_pond::observe(state_,obs);
        if(selector_actor_) cudarl::demos::primary_pond_food(obs,selected_food_);
        actor_->forward(obs,1);
        cudarl::deep_pond::advance(state_,actor_->output());
    }
    return get_state();
}
Dictionary DeepPondLab::get_state() const {
    Dictionary d;d["ready"]=static_cast<bool>(actor_);if(!actor_) return d;
    const auto& p=state_.body.position;const auto& q=state_.body.orientation;
    d["position"]=Vector3(p.x,p.y,p.z);d["orientation"]=Quaternion(q.x,q.y,q.z,q.w);
    PackedVector3Array foods;
    for(const auto& f:state_.food) foods.push_back(Vector3(f.x,f.y,f.z));
    d["foods"]=foods;d["food"]=foods[0];d["eaten_slot"]=state_.last_eaten;
    d["eaten"]=state_.eaten;d["elapsed"]=state_.steps*cudarl::deep::dt;
    d["selected_food"]=selected_food_;
    d["seconds_since_food"]=state_.since_food*cudarl::deep::dt;
    d["depth_cm"]=-p.y*4.;
    d["speed_cm_s"]=100.*cudarl::deep::norm({state_.body.velocity[0],state_.body.velocity[1],state_.body.velocity[2]});
    const auto shape=cudarl::deep::shape(state_.body);Array links;
    for(const auto& link:shape.link) {
        Dictionary item;const auto& r=link.q;const auto c=link.center/cudarl::deep::scale;
        item["position"]=Vector3(c.x,c.y,c.z);item["orientation"]=Quaternion(r.x,r.y,r.z,r.w);links.push_back(item);
    }
    d["links"]=links;return d;
}
Dictionary DeepPondLab::get_stats() {
    collect();Dictionary d;
    d["validated_controller"]=static_cast<bool>(selector_actor_);
    if(selector_actor_) {
        d["training_ready"]=true;d["paused"]=true;d["resumed"]=false;d["loaded_update"]=controller_.update;d["saved_update"]=controller_.update;
        d["transferred"]=false;d["food_count"]=3;d["policy_inputs"]=33;d["watching_best"]=true;d["save_error"]=String();
        d["policy_update"]=controller_.motor_update;d["selector_update"]=controller_.update;d["decision_steps"]=controller_.decision_steps;d["update"]=controller_.update;
        d["env_steps"]=int64_t(0);d["update_ms"]=0.;d["training_seconds"]=0.;d["background_swimmers"]=0;
        d["current_food_per_minute"]=controller_.food_per_minute();d["best_food_per_minute"]=controller_.food_per_minute();d["best_update"]=controller_.update;
        d["evaluation_count"]=1;d["checkpoint_path"]=controller_path_;return d;
    }
    const auto save=session_?session_->checkpoint_status():cudarl::checkpoint::Status{};
    d["training_ready"]=session_ && session_->ready();d["paused"]=session_ && session_->paused();
    d["resumed"]=save.resumed;d["loaded_update"]=save.loaded_update;d["saved_update"]=save.saved_update;
    d["transferred"]=save.imported;d["food_count"]=3;d["policy_inputs"]=cudarl::deep_pond::kObs;
    d["watching_best"]=watching_best_;
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
Array DeepPondLab::get_evaluation_history(int from) {
    collect();Array a;
    if(selector_actor_) {
        if(from==0) {Dictionary d;d["update"]=controller_.update;d["food_per_minute"]=controller_.food_per_minute();a.push_back(d);}return a;
    }
    for(size_t i=static_cast<size_t>(std::max(0,from));i<evaluations_.size();++i) {
        const auto& s=evaluations_[i];Dictionary d;d["update"]=s.evaluation_update;
        d["food_per_minute"]=s.evaluation_foods*60./(s.evaluation_count*s.evaluation_seconds);a.push_back(d);
    }
    return a;
}
}
