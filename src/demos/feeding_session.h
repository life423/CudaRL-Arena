#ifndef CUDARL_FEEDING_SESSION_H
#define CUDARL_FEEDING_SESSION_H
#include "envs/swimmer/feeding.h"
#include <memory>
#include <string>
namespace cudarl::demos {
struct FeedingStats {
    int update=0,adopted_update=0,applied_update=0,saved_update=-1,loaded_update=0,context_version=0,trained_context=0,worlds=0,evaluations=0;
    double update_ms=0,original_score=-1,candidate_score=-1,adopted_score=-1;
    bool ready=false,resumed=false,training=false,paused=false,learner_changed=false;
    std::string error,save_error;
    std::string backend="CUDA";
    bool arch_mode=false;
    int input_count=feeding::kObs,parent_update=0,parent_selector_update=0,arch_contacts=0,arch_passages=0,arch_clean_passages=0;
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
};
std::unique_ptr<FeedingSession> create_feeding_session(const std::string& policy,const std::string& checkpoint,int worlds=1024);
std::unique_ptr<FeedingSession> create_cpu_feeding_session(const std::string& policy,const std::string& checkpoint,int worlds=16);
std::unique_ptr<FeedingSession> create_arch_session(const std::string& source_full_save,const std::string& checkpoint,int worlds=1024);
}
#endif
