#include "deep_pond_lab.h"
#include <exception>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/classes/project_settings.hpp>
#include <godot_cpp/variant/vector3.hpp>
#include <godot_cpp/variant/quaternion.hpp>
#include <godot_cpp/variant/packed_vector3_array.hpp>
#include <godot_cpp/variant/packed_int64_array.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#if CUDARL_WITH_CUDA
#include "cuda/device.h"
#endif
namespace godot {
void DeepPondLab::_bind_methods() {
    ClassDB::bind_method(D_METHOD("start_training","environments","seed","checkpoint","source"),&DeepPondLab::start_training,
        DEFVAL(1024),DEFVAL(7),DEFVAL("user://pond-policies/deep-pond-v1/training.bin"),DEFVAL("user://pond-policies/deep-v1/training.bin"));
    ClassDB::bind_method(D_METHOD("stop"),&DeepPondLab::stop);
    ClassDB::bind_method(D_METHOD("watch_controller","path","backend"),&DeepPondLab::watch_controller,DEFVAL("auto"));
    ClassDB::bind_method(D_METHOD("watch_progress_controller","path","backend"),&DeepPondLab::watch_progress_controller,DEFVAL("auto"));
    ClassDB::bind_method(D_METHOD("set_untrained","enabled"),&DeepPondLab::set_untrained);
    ClassDB::bind_method(D_METHOD("set_gentle_current","enabled","drifting_food"),&DeepPondLab::set_gentle_current,DEFVAL(false));
    ClassDB::bind_method(D_METHOD("set_steady_current","speed_cm_s","drifting_food"),&DeepPondLab::set_steady_current,DEFVAL(false));
    ClassDB::bind_method(D_METHOD("start_feeding_learning","policy","checkpoint","environments"),&DeepPondLab::start_feeding_learning,DEFVAL(1024));
    ClassDB::bind_method(D_METHOD("drop_food","position"),&DeepPondLab::drop_food);
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
#if CUDARL_WITH_CUDA
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
#else
    (void)environments;(void)seed;(void)checkpoint;(void)source;
    error_="This portable player runs the trained controller. CUDA training is available in the developer build.";return false;
#endif
}
void DeepPondLab::stop() {
#if CUDARL_WITH_CUDA
    if(feeding_) feeding_->stop();feeding_.reset();
    if(session_) session_->stop();session_.reset();actor_.reset();history_.clear();evaluations_.clear();
    policy_version_=0;best_version_=0;policy_update_=0;training_ms_=0;best_={};applied_best_update_=-1;watching_best_=true;
#endif
    playback_.reset();actor_.reset();controller_={};controller_path_=String();selected_food_=-1;policy_update_=0;
    showcase_untrained_=false;controller_backend_="auto";
    progress_controller_={};use_progress_controller_=false;
}
bool DeepPondLab::watch_controller(String path,String backend) {
    stop();error_=String();
    try {
        controller_=cudarl::rl::SelectorPolicy::load(ProjectSettings::get_singleton()->globalize_path(path).utf8().get_data());
        const auto module=ProjectSettings::get_singleton()->globalize_path("res://addons/cudarl/bin/cudarl_pond_cuda.dll").utf8();
        playback_=cudarl::demos::create_pond_playback(controller_,backend.utf8().get_data(),module.get_data());
        UtilityFunctions::print("POND_PLAYBACK backend=",playback_->backend()," detail=",playback_->detail().c_str());
        state_=playback_->frame().state;selected_food_=playback_->frame().selected;
        policy_update_=controller_.motor_update;controller_path_=path;controller_backend_=backend;return true;
    } catch(const std::exception& e) {stop();error_=e.what();return false;}
}
bool DeepPondLab::set_untrained(bool enabled) {
    if(!playback_) return false;
    if(enabled==showcase_untrained_) return true;
    try {
        const auto module=ProjectSettings::get_singleton()->globalize_path("res://addons/cudarl/bin/cudarl_pond_cuda.dll").utf8();
        std::unique_ptr<cudarl::demos::PondPlayback> next;
        if(use_progress_controller_) {
            auto policy=progress_controller_;
            if(enabled) {
                policy.baseline=cudarl::rl::initial_pond_policy();policy.update=0;
                policy.actor=cudarl::nn::orthogonal_params(cudarl::rl::progress_selector_spec(),8,{std::sqrt(2.f),std::sqrt(2.f),.01f});
            }
            next=cudarl::demos::create_pond_playback(policy,controller_backend_.utf8().get_data(),module.get_data());
        } else next=cudarl::demos::create_pond_playback(enabled?cudarl::rl::initial_pond_policy():controller_,
            controller_backend_.utf8().get_data(),module.get_data());
        next->set_food_drift(playback_->frame().food_drift);
        next->set_current(playback_->current());
        playback_=std::move(next);showcase_untrained_=enabled;error_=String();
        state_=playback_->frame().state;selected_food_=playback_->frame().selected;
        return true;
    } catch(const std::exception& e) {error_=e.what();return false;}
}
bool DeepPondLab::watch_progress_controller(String path,String backend) {
    stop();error_=String();
    try {
        progress_controller_=cudarl::rl::ProgressPolicy::load(ProjectSettings::get_singleton()->globalize_path(path).utf8().get_data());
        controller_=progress_controller_.baseline;controller_.update=progress_controller_.update;
        const auto module=ProjectSettings::get_singleton()->globalize_path("res://addons/cudarl/bin/cudarl_pond_cuda.dll").utf8();
        playback_=cudarl::demos::create_pond_playback(progress_controller_,backend.utf8().get_data(),module.get_data());
        use_progress_controller_=true;state_=playback_->frame().state;selected_food_=playback_->frame().selected;
        policy_update_=controller_.motor_update;controller_path_=path;controller_backend_=backend;
        UtilityFunctions::print("POND_LIVE_CANDIDATE selector=",controller_.update," backend=",playback_->backend()," detail=",playback_->detail().c_str());
        return true;
    } catch(const std::exception& e) {stop();error_=e.what();return false;}
}
bool DeepPondLab::start_feeding_learning(String policy,String checkpoint,int environments) {
    stop();error_=String();
#if CUDARL_WITH_CUDA
    try {
        const auto source=cudarl::rl::ProgressPolicy::load(ProjectSettings::get_singleton()->globalize_path(policy).utf8().get_data());
        controller_=source.baseline;controller_.update=source.update;controller_path_=policy;
        feeding_=cudarl::demos::create_feeding_session(ProjectSettings::get_singleton()->globalize_path(policy).utf8().get_data(),
            ProjectSettings::get_singleton()->globalize_path(checkpoint).utf8().get_data(),environments);
        UtilityFunctions::print("POND_FEEDING CUDA live + PPO worlds=",environments," source selector=",source.update);return true;
    } catch(const std::exception& e) {feeding_.reset();error_=e.what();return false;}
#else
    (void)policy;(void)checkpoint;(void)environments;error_="Sinking-food learning requires the native CUDA training build. The saved reference pond remains available.";return false;
#endif
}
int DeepPondLab::drop_food(Vector3 position) {
#if CUDARL_WITH_CUDA
    if(feeding_) {
        try {return feeding_->drop({static_cast<float>(position.x),static_cast<float>(position.y),static_cast<float>(position.z)});}
        catch(const std::exception& e) {error_=e.what();}
    }
#else
    (void)position;
#endif
    return -1;
}
bool DeepPondLab::set_gentle_current(bool enabled,bool drifting_food) {
    return set_steady_current(enabled?.5:0.,drifting_food);
}
bool DeepPondLab::set_steady_current(double speed_cm_s,bool drifting_food) {
    if(!std::isfinite(speed_cm_s) || speed_cm_s<0 || speed_cm_s>.75) {
        error_="Live steady current must be between 0 and 0.75 cm/s";return false;
    }
#if CUDARL_WITH_CUDA
    if(feeding_) {
        try {feeding_->set_current({static_cast<float>(speed_cm_s*.01),0,0});error_=String();return true;}
        catch(const std::exception& e) {error_=e.what();return false;}
    }
#endif
    if(!playback_) return false;
    const auto previous_current=playback_->current();const bool previous_drift=playback_->frame().food_drift;
    try {
        playback_->set_food_drift(speed_cm_s>0 && drifting_food);
        playback_->set_current({static_cast<float>(speed_cm_s*.01),0,0});error_=String();return true;
    } catch(const std::exception& e) {
        error_=e.what();
        try {playback_->set_food_drift(previous_drift);playback_->set_current(previous_current);} catch(const std::exception&) {}
        return false;
    }
}
void DeepPondLab::set_paused(bool paused) {
#if CUDARL_WITH_CUDA
    if(feeding_) feeding_->set_paused(paused);
    if(session_) session_->set_paused(paused);
#else
    (void)paused;
#endif
}
void DeepPondLab::set_updates_per_second(double speed) {
#if CUDARL_WITH_CUDA
    if(session_) session_->set_updates_per_second(speed);
#else
    (void)speed;
#endif
}
void DeepPondLab::save_training_now() {
#if CUDARL_WITH_CUDA
    if(feeding_) feeding_->request_save();
    if(session_) session_->request_save();
#endif
}
void DeepPondLab::set_watch_best(bool enabled) {
#if CUDARL_WITH_CUDA
    watching_best_=enabled;policy_version_=0;applied_best_update_=-1;
#else
    (void)enabled;
#endif
}
String DeepPondLab::get_error() const {
#if CUDARL_WITH_CUDA
    return !error_.is_empty()?error_:(session_?String(session_->error().c_str()):String());
#else
    return error_;
#endif
}
void DeepPondLab::collect() {
#if CUDARL_WITH_CUDA
    if(!session_) return;const auto first=history_.size();session_->history_since(first,history_);
    for(size_t i=first;i<history_.size();++i) {
        const auto& s=history_[i];training_ms_+=s.update_ms;
        if(s.evaluation_count>0 && (evaluations_.empty() || s.evaluation_update>evaluations_.back().evaluation_update)) evaluations_.push_back(s);
    }
    session_->latest_best_policy(best_version_,best_);
#endif
}
Dictionary DeepPondLab::step() {
#if CUDARL_WITH_CUDA
    if(feeding_) {try {feeding_->step();} catch(const std::exception& e) {error_=e.what();}return get_state();}
#endif
    if(playback_) {
        try {const auto& next=playback_->step();state_=next.state;selected_food_=next.selected;}
        catch(const std::exception& e) {error_=e.what();}
        return get_state();
    }
#if CUDARL_WITH_CUDA
    if(actor_) {
        std::vector<float> parameters;
        collect();
        if(watching_best_ && best_.valid()) {
            if(applied_best_update_!=best_.update) {actor_->params()=best_.parameters;policy_update_=best_.update;applied_best_update_=best_.update;}
        } else if(session_ && session_->latest_policy(policy_version_,parameters,&policy_update_)) actor_->params()=std::move(parameters);
        float obs[cudarl::deep_pond::kObs];cudarl::deep_pond::observe(state_,obs);
        actor_->forward(obs,1);
        cudarl::deep_pond::advance(state_,actor_->output());
    }
#endif
    return get_state();
}
Dictionary DeepPondLab::get_state() const {
#if CUDARL_WITH_CUDA
    if(feeding_) {
        const auto& s=feeding_->state();const auto& p=s.body.position;const auto& q=s.body.orientation;Dictionary d;
        d["ready"]=true;d["position"]=Vector3(p.x,p.y,p.z);d["orientation"]=Quaternion(q.x,q.y,q.z,q.w);
        PackedVector3Array foods,velocities;PackedInt64Array identities;
        for(const auto& food:s.pellets) if(food.active) {foods.push_back(Vector3(food.position.x,food.position.y,food.position.z));velocities.push_back(Vector3(food.velocity.x,food.velocity.y,food.velocity.z));identities.push_back(food.id);}
        d["foods"]=foods;d["pellet_velocities"]=velocities;d["pellet_ids"]=identities;d["food"]=foods.is_empty()?Vector3(p.x,p.y,p.z):foods[0];d["eaten_slot"]=-1;
        d["eaten"]=s.eaten;d["elapsed"]=s.steps*cudarl::deep::dt;d["selected_food"]=s.target;
        d["seconds_since_food"]=s.since_food*cudarl::deep::dt;d["depth_cm"]=-p.y*4.;d["speed_cm_s"]=100.*cudarl::deep::norm({s.body.velocity[0],s.body.velocity[1],s.body.velocity[2]});
        const auto geometry=cudarl::deep::shape(s.body);Array links;
        for(const auto& link:geometry.link) {Dictionary item;const auto c=link.center/cudarl::deep::scale;const auto& r=link.q;item["position"]=Vector3(c.x,c.y,c.z);item["orientation"]=Quaternion(r.x,r.y,r.z,r.w);links.push_back(item);}
        d["links"]=links;return d;
    }
#endif
    Dictionary d;d["ready"]=static_cast<bool>(actor_) || static_cast<bool>(playback_);if(!actor_ && !playback_) return d;
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
#if CUDARL_WITH_CUDA
    if(feeding_) {
        const auto s=feeding_->stats();const auto current=feeding_->current();
        d["feeding_mode"]=true;d["playback_mode"]=true;d["showcase_untrained"]=false;d["validated_controller"]=false;d["progress_controller"]=true;
        d["water_current_cm_s"]=100.*cudarl::deep::norm(current);d["water_current"]=Vector3(current.x,current.y,current.z);d["food_drifting"]=cudarl::deep::norm(current)>0;
        d["compute_backend"]="CUDA";d["compute_detail"]="Native CUDA physics/inference and concurrent PPO";d["training_ready"]=true;d["paused"]=s.paused;
        d["resumed"]=s.resumed;d["loaded_update"]=s.loaded_update;d["saved_update"]=s.saved_update;d["save_error"]=s.save_error.c_str();d["transferred"]=false;
        d["food_count"]=cudarl::feeding::count(feeding_->state());d["food_capacity"]=cudarl::feeding::capacity;d["policy_inputs"]=cudarl::feeding::kObs;
        d["watching_best"]=true;d["policy_update"]=controller_.motor_update;d["selector_update"]=controller_.update;d["decision_steps"]=20;d["update"]=s.update;
        d["env_steps"]=static_cast<int64_t>(s.update)*s.worlds*64;d["update_ms"]=s.update_ms;d["training_seconds"]=0.;d["background_swimmers"]=s.worlds;
        d["current_food_per_minute"]=s.candidate_score;d["best_food_per_minute"]=s.adopted_score;d["best_update"]=s.adopted_update;d["evaluation_count"]=s.evaluations;d["checkpoint_path"]=controller_path_;
        d["learning_ready"]=s.ready;d["learning_active"]=s.training && !s.paused;d["learning_error"]=s.error.c_str();d["adopted_update"]=s.applied_update;d["approved_update"]=s.adopted_update;d["learner_weights_changed"]=s.learner_changed;
        d["original_food_per_minute"]=s.original_score;d["context_version"]=s.context_version;d["trained_context_version"]=s.trained_context;return d;
    }
#endif
    d["playback_mode"]=static_cast<bool>(playback_);
    d["showcase_untrained"]=showcase_untrained_;
    d["validated_controller"]=static_cast<bool>(playback_) && !showcase_untrained_ && !use_progress_controller_;
    d["progress_controller"]=use_progress_controller_;
    if(playback_) {
        const auto current=playback_->current();
        d["water_current_cm_s"]=100.*cudarl::deep::norm(current);
        d["water_current"]=Vector3(current.x,current.y,current.z);
        d["food_drifting"]=playback_->frame().food_drift;
        d["compute_backend"]=playback_->backend();d["compute_detail"]=playback_->detail().c_str();
        d["training_ready"]=true;d["paused"]=true;d["resumed"]=false;d["loaded_update"]=controller_.update;d["saved_update"]=controller_.update;
        d["transferred"]=false;d["food_count"]=3;d["policy_inputs"]=33;d["watching_best"]=true;d["save_error"]=String();
        d["policy_update"]=controller_.motor_update;d["selector_update"]=controller_.update;d["decision_steps"]=controller_.decision_steps;d["update"]=controller_.update;
        d["env_steps"]=int64_t(0);d["update_ms"]=0.;d["training_seconds"]=0.;d["background_swimmers"]=0;
        d["current_food_per_minute"]=use_progress_controller_?0.:controller_.food_per_minute();d["best_food_per_minute"]=use_progress_controller_?0.:controller_.food_per_minute();d["best_update"]=controller_.update;
        d["evaluation_count"]=use_progress_controller_?0:1;d["checkpoint_path"]=controller_path_;return d;
    }
#if CUDARL_WITH_CUDA
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
#else
    return d;
#endif
}
Array DeepPondLab::get_evaluation_history(int from) {
    collect();Array a;
    if(playback_) {
        if(from==0 && !use_progress_controller_) {Dictionary d;d["update"]=controller_.update;d["food_per_minute"]=controller_.food_per_minute();a.push_back(d);}return a;
    }
#if CUDARL_WITH_CUDA
    for(size_t i=static_cast<size_t>(std::max(0,from));i<evaluations_.size();++i) {
        const auto& s=evaluations_[i];Dictionary d;d["update"]=s.evaluation_update;
        d["food_per_minute"]=s.evaluation_foods*60./(s.evaluation_count*s.evaluation_seconds);a.push_back(d);
    }
#endif
    return a;
}
}
