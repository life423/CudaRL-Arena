#ifndef CUDARL_FEEDING_SESSION_H
#define CUDARL_FEEDING_SESSION_H
#include "envs/swimmer/feeding.h"
#include <memory>
#include <string>
namespace cudarl::demos {
struct PredatorInfo {
    bool enabled=false,warning=false,active=false;
    deep::V position{},shadow{},direction{},goal{};
    float strength=0,exposure=1;
    float held_seconds=0,longest_safe_seconds=0,clearance=0,goal_distance=0;
    bool cover_goal=false;
};
struct FeedingStats {
    int update=0,adopted_update=0,applied_update=0,saved_update=-1,loaded_update=0,context_version=0,trained_context=0,worlds=0,evaluations=0;
    double update_ms=0,original_score=-1,candidate_score=-1,adopted_score=-1;
    bool ready=false,resumed=false,training=false,paused=false,learner_changed=false;
    std::string error,save_error;
    std::string backend="CUDA";
    bool arch_mode=false;
    int input_count=feeding::kObs,parent_update=0,parent_selector_update=0,arch_contacts=0,arch_passages=0,arch_clean_passages=0;
    int evaluated_context=-1;
    double arch_calm_score=-1,arch_calm_reference_score=-1,arch_current_score=-1;
    int layout_id=0,parent_stage_update=0;
    bool imported_full_state=false;
    std::string habitat_name="Rock arch pond";
    double habitat_score=-1;
    double cove_retention_score=-1,cove_retention_reference_score=-1;
    bool shelter_mode=false;
    bool hold_practice_mode=false;
    double shelter_exposure=-1,shelter_reference_exposure=-1,calm_retention_score=-1;
};
class FeedingSession {
public:
    virtual ~FeedingSession()=default;
    virtual feeding::State step()=0;
    virtual const feeding::State& state() const=0;
    virtual int drop(deep::V position)=0;
    virtual void set_current(deep::V current)=0;
    virtual deep::V current() const=0;
    virtual FeedingStats stats() const=0;
    virtual void request_save()=0;
    virtual void set_paused(bool paused)=0;
    virtual void stop()=0;
    virtual PredatorInfo predator_info() const {return {};}
    virtual bool trigger_predator() {return false;}
};
std::unique_ptr<FeedingSession> create_feeding_session(const std::string& policy,const std::string& checkpoint,int worlds=1024);
std::unique_ptr<FeedingSession> create_cpu_feeding_session(const std::string& policy,const std::string& checkpoint,int worlds=16);
std::unique_ptr<FeedingSession> create_arch_session(const std::string& source_full_save,const std::string& checkpoint,int worlds=1024);
std::unique_ptr<FeedingSession> create_cove_session(const std::string& source_arch_save,const std::string& checkpoint,int worlds=1024);
std::unique_ptr<FeedingSession> create_boulder_session(const std::string& source_cove_save,const std::string& checkpoint,int worlds=1024);
std::unique_ptr<FeedingSession> create_shelter_session(const std::string& source_boulder_save,const std::string& checkpoint,int worlds=1024);
std::unique_ptr<FeedingSession> create_hold_session(const std::string& source_motor_run,const std::string& checkpoint,int worlds=1024);
}
#endif
