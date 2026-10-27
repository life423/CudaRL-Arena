#include "tadpole_lab.h"
#include <exception>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/packed_vector2_array.hpp>
#include <godot_cpp/classes/dir_access.hpp>
#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/classes/json.hpp>
#include <godot_cpp/classes/project_settings.hpp>
#include <godot_cpp/classes/time.hpp>
#include "cuda/device.h"
namespace godot {
void TadpoleLab::_bind_methods() {
    ClassDB::bind_method(D_METHOD("watch_policy","weights","seed"),&TadpoleLab::watch_policy,DEFVAL(1));
    ClassDB::bind_method(D_METHOD("watch_foraging_checkpoint","path","seed"),&TadpoleLab::watch_foraging_checkpoint,DEFVAL(1));
    ClassDB::bind_method(D_METHOD("step"),&TadpoleLab::step);
    ClassDB::bind_method(D_METHOD("get_state"),&TadpoleLab::get_state);
    ClassDB::bind_method(D_METHOD("reset","seed"),&TadpoleLab::reset,DEFVAL(1));
    ClassDB::bind_method(D_METHOD("move_food"),&TadpoleLab::move_food);
    ClassDB::bind_method(D_METHOD("get_error"),&TadpoleLab::get_error);
    ClassDB::bind_method(D_METHOD("start_training","num_envs","seed","updates_per_second","checkpoint_path","shuffle_inputs","resume_training"),&TadpoleLab::start_training,DEFVAL(1024),DEFVAL(1),DEFVAL(10.0),DEFVAL("user://pond-policies/regional-v1/best.json"),DEFVAL(false),DEFVAL(true));
    ClassDB::bind_method(D_METHOD("save_training_now"),&TadpoleLab::save_training_now);
    ClassDB::bind_method(D_METHOD("set_watch_best","enabled"),&TadpoleLab::set_watch_best);
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
bool TadpoleLab::watch_foraging_checkpoint(String path,int seed) {
    stop();error_=String();
    checkpoint_path_=ProjectSettings::get_singleton()->globalize_path(path);load_best();
    if(!best_.valid()) {error_=checkpoint_error_.is_empty()?String("Reference checkpoint could not be opened."):checkpoint_error_;return false;}
    try {
        foraging_=std::make_unique<cudarl::demos::ForagingPond>(cudarl::swimmer::body_preset(8));
        foraging_->set_policy(best_.parameters);foraging_->reset(static_cast<uint32_t>(seed));applied_update_=best_.update;
        return true;
    } catch(const std::exception& e) {error_=e.what();return false;}
}
bool TadpoleLab::start_training(int num_envs,int seed,double updates_per_second,String checkpoint_path,bool shuffle_inputs,bool resume_training) {
    stop();error_=String();
    if(num_envs<=0 || num_envs>32768) {error_="Invalid background swimmer count";return false;}
    const auto device=cudarl::probe_device();
    if(!device.available) {error_=String("CUDA unavailable: ")+device.error.c_str();return false;}
    try {
        checkpoint_path_=checkpoint_path.is_empty()?String():ProjectSettings::get_singleton()->globalize_path(checkpoint_path);
        load_best();
        starting_best_score_=best_.food_per_minute();
        cudarl::rl::SwimmerConfig cfg;cfg.num_envs=num_envs;cfg.seed=static_cast<uint32_t>(seed);
        cfg.total_updates=0; // Keep learning until the viewer explicitly stops the session.
        cfg.evaluation_every=25;
        cfg.body=cudarl::swimmer::body_preset(8);
        cfg.shuffle_food_inputs=shuffle_inputs;
        cfg.training_checkpoint_path=checkpoint_path_.is_empty()?std::string():std::string((checkpoint_path_+".training").utf8().get_data());
        cfg.resume_training=resume_training;
        if(resume_training && best_.valid() && best_.shuffled_inputs==shuffle_inputs && best_.training_seed==cfg.seed) cfg.import_policy=best_;
        foraging_=std::make_unique<cudarl::demos::ForagingPond>(cfg.body);
        foraging_->set_policy(cfg.import_policy.valid()?cfg.import_policy.parameters:cudarl::rl::ForagingTrainer::initial_actor_params(cfg.seed));
        foraging_->reset(cfg.seed);
        session_=std::make_unique<cudarl::demos::TrainingSession<cudarl::rl::ForagingTrainer>>(cfg,updates_per_second,true);
        return true;
    } catch(const std::exception& e) {stop();error_=e.what();return false;}
}
void TadpoleLab::set_paused(bool paused) {if(session_) session_->set_paused(paused);}
void TadpoleLab::save_training_now() {if(session_) session_->request_save();}
void TadpoleLab::set_updates_per_second(double updates_per_second) {if(session_) session_->set_updates_per_second(updates_per_second);}
void TadpoleLab::stop() {
    // Join before the last save: an in-flight update may publish a new winner.
    if(session_) {session_->stop();collect_stats();}
    session_.reset();pond_.reset();foraging_.reset();history_.clear();policy_version_=0;applied_update_=0;
    evaluations_.clear();training_ms_=0;
    session_best_version_=0;best_revision_=0;applied_best_revision_=0;watching_best_=false;
    best_={};best_saved_=false;
}
String TadpoleLab::get_error() const {
    if(!error_.is_empty()) return error_;
    return session_?String(session_->error().c_str()):String();
}
void TadpoleLab::collect_stats() {
    const size_t previous=history_.size();
    if(session_) session_->history_since(previous,history_);
    bool new_evaluation=false;
    for(size_t i=previous;i<history_.size();++i) {
        const auto& stats=history_[i];
        training_ms_+=stats.update_ms;
        if(stats.evaluation_count>0 && (evaluations_.empty() || stats.evaluation_update>evaluations_.back().evaluation_update))
            {evaluations_.push_back(stats);new_evaluation=true;}
    }
    cudarl::rl::EvaluatedPolicy candidate;
    if(session_ && session_->latest_best_policy(session_best_version_,candidate) &&
       (!best_.valid() || candidate.food_per_minute()>best_.food_per_minute())) {
        best_=std::move(candidate);best_saved_=false;++best_revision_;save_best();
    } else if(best_.valid() && !best_saved_ && new_evaluation) save_best();
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
        const double run_best=stats.best_evaluation_foods<0?0.0:stats.best_evaluation_foods*60.0/(stats.evaluation_count*stats.evaluation_seconds);
        point["best_food_per_minute"]=std::max(starting_best_score_,run_best);
        result.push_back(point);
    }
    return result;
}
Dictionary TadpoleLab::get_stats() {
    collect_stats();
    Dictionary d;
    d["training"]=static_cast<bool>(session_);
    const auto persistence=session_?session_->checkpoint_status():cudarl::checkpoint::Status{};
    d["training_ready"]=session_ && session_->ready();
    d["resumed_training"]=persistence.resumed;d["imported_best"]=persistence.imported;
    d["training_saved_update"]=persistence.saved_update;d["training_loaded_update"]=persistence.loaded_update;
    d["training_save_error"]=String(persistence.error.c_str());
    d["training_save_path"]=session_?String(session_->config().training_checkpoint_path.c_str()):String();
    d["shuffled_inputs"]=session_ && session_->config().shuffle_food_inputs;
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
    d["best_ready"]=best_.valid();d["best_saved"]=best_saved_;
    d["best_food_per_minute"]=best_.food_per_minute();d["best_update"]=best_.update;
    d["checkpoint_path"]=checkpoint_path_;d["checkpoint_error"]=checkpoint_error_;
    d["watching_best"]=watching_best_;
    d["evaluation_update"]=evaluations_.empty()?-1:evaluations_.back().evaluation_update;
    d["current_food_per_minute"]=evaluations_.empty()?0.0:evaluations_.back().evaluation_foods*60.0/(evaluations_.back().evaluation_count*evaluations_.back().evaluation_seconds);
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
        if(session_ && foraging_ && watching_best_) {
            collect_stats();
            if(best_.valid() && applied_best_revision_!=best_revision_) {
                foraging_->set_policy(best_.parameters);applied_update_=best_.update;applied_best_revision_=best_revision_;
            }
        } else if(session_ && foraging_) {
            std::vector<float> parameters;
            if(session_->latest_policy(policy_version_,parameters,&applied_update_)) {
                foraging_->set_policy(parameters);
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
bool TadpoleLab::set_watch_best(bool enabled) {
    collect_stats();
    if(!foraging_ || (enabled && !best_.valid())) return false;
    watching_best_=enabled;policy_version_=0;applied_best_revision_=0;
    if(!enabled && session_) {
        std::vector<float> parameters;
        if(session_->latest_policy(policy_version_,parameters,&applied_update_)) {
            foraging_->set_policy(parameters);
        } else {
            foraging_->set_policy(cudarl::rl::ForagingTrainer::initial_actor_params(session_->config().seed));applied_update_=0;
        }
    }
    return true;
}

void TadpoleLab::load_best() {
    best_={};best_saved_=false;checkpoint_error_=String();checkpoint_blocked_=false;
    if(checkpoint_path_.is_empty() || !FileAccess::file_exists(checkpoint_path_)) return;
    const auto invalid=[&](const String& reason) {
        checkpoint_error_=String("Saved best was kept but could not be loaded: ")+reason;
        checkpoint_blocked_=true;
    };
    Ref<FileAccess> file=FileAccess::open(checkpoint_path_,FileAccess::READ);
    if(file.is_null()) {invalid("file could not be opened");return;}
    Ref<JSON> json;json.instantiate();
    if(json->parse(file->get_as_text())!=OK || json->get_data().get_type()!=Variant::DICTIONARY) {
        invalid("invalid JSON");return;
    }
    const Dictionary data=json->get_data();
    if(String(data.get("protocol",String()))!="three-food-regional-inertial-v1" ||
       static_cast<double>(data.get("version",0))!=1.0 || static_cast<double>(data.get("inputs",0))!=13.0 ||
       static_cast<double>(data.get("hidden",0))!=32.0 || static_cast<double>(data.get("outputs",0))!=2.0 ||
       static_cast<double>(data.get("body_id",0))!=8.0 || static_cast<double>(data.get("count",0))!=64.0 ||
       static_cast<double>(data.get("steps",0))!=1200.0 || static_cast<double>(data.get("evaluation_seed",0))!=static_cast<double>(0x51A7E123u) ||
       String(data.get("inference",String()))!="deterministic_mean") {
        invalid("incompatible policy or evaluation protocol");return;
    }
    const double foods=static_cast<double>(data.get("foods",-1)),update=static_cast<double>(data.get("update",-1));
    if(!std::isfinite(foods) || foods<0 || foods>64*1200 || std::floor(foods)!=foods ||
       !std::isfinite(update) || update<0 || update>2147483647 || std::floor(update)!=update ||
       data.get("parameters",Variant()).get_type()!=Variant::ARRAY) {invalid("invalid score or weights");return;}
    const Array parameters=data["parameters"];
    if(parameters.size()!=cudarl::rl::swimmer_specification(2,13).num_params()) {invalid("wrong weight count");return;}
    std::vector<float> weights;weights.reserve(static_cast<size_t>(parameters.size()));
    for(int64_t i=0;i<parameters.size();++i) {
        const Variant value=parameters[i];
        if(value.get_type()!=Variant::FLOAT && value.get_type()!=Variant::INT) {invalid("invalid weight");return;}
        const float weight=static_cast<float>(static_cast<double>(value));
        if(!std::isfinite(weight)) {invalid("non-finite weight");return;}weights.push_back(weight);
    }
    best_={static_cast<int>(update),static_cast<int>(foods),64,1200,0x51A7E123u,
           static_cast<uint32_t>(static_cast<int64_t>(data.get("training_seed",1))),std::move(weights),
           String(data.get("training_variant",String("canonical")))=="shuffled"};
    best_saved_=true;++best_revision_;
}
void TadpoleLab::save_best() {
    if(checkpoint_path_.is_empty() || checkpoint_blocked_ || !best_.valid()) return;
    Dictionary data;
    data["version"]=1;data["protocol"]="three-food-regional-inertial-v1";
    data["inputs"]=13;data["hidden"]=32;data["outputs"]=2;data["body_id"]=8;
    data["update"]=best_.update;data["foods"]=best_.foods;data["count"]=best_.count;data["steps"]=best_.steps;
    data["evaluation_seed"]=static_cast<int64_t>(best_.evaluation_seed);data["training_seed"]=static_cast<int64_t>(best_.training_seed);
    data["food_per_minute"]=best_.food_per_minute();data["inference"]="deterministic_mean";
    data["training_variant"]=best_.shuffled_inputs?"shuffled":"canonical";
    data["saved_at_utc"]=Time::get_singleton()->get_datetime_string_from_system(true,true);
    Array parameters;for(float weight:best_.parameters) parameters.push_back(weight);data["parameters"]=parameters;
    if(DirAccess::make_dir_recursive_absolute(checkpoint_path_.get_base_dir())!=OK) {
        checkpoint_error_="Could not create the best-policy folder.";return;
    }
    const String temporary=checkpoint_path_+".tmp";
    Ref<FileAccess> file=FileAccess::open(temporary,FileAccess::WRITE);
    if(file.is_null()) {checkpoint_error_="Could not write the best-policy file.";return;}
    file->store_string(JSON::stringify(data,"\t",true,true));file->flush();
    const Error error=file->get_error();file.unref();
    if(error!=OK || DirAccess::rename_absolute(temporary,checkpoint_path_)!=OK) {
        checkpoint_error_="Could not replace the saved best policy. The previous file was kept.";return;
    }
    best_saved_=true;checkpoint_error_=String();
}
}
