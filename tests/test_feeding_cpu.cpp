#include "rl/feeding_cpu.h"
#include "demos/feeding_session.h"
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <thread>
int main(int argc,char** argv) {
    int checks=0;
    auto check=[&](bool ok,const char* message) {++checks;if(!ok) throw std::runtime_error(message);};
    try {
        if(argc!=2) throw std::runtime_error("Expected source progress policy");
        const auto source=cudarl::rl::ProgressPolicy::load(argv[1]);
        namespace fd=cudarl::feeding;namespace rl=cudarl::rl;
        fd::CpuBatch worlds(1,7,source.actor);
        worlds.states[0]=fd::initial(7,0,0,{},false);worlds.prepare();
        float action[4]={.1f,-.2f,.3f,-.1f};
        for(int t=0;t<100;++t) {
            auto expected=worlds.states[0];fd::advance(expected,action,{.005f,0,0});
            worlds.context.current={.005f,0,0};worlds.step(action,nullptr,nullptr,nullptr,false);
            check(cudarl::deep::norm(expected.body.position-worlds.states[0].body.position)==0,"CPU batch changed shared physical integration");
            for(int i=0;i<fd::capacity;++i) check(cudarl::deep::norm(expected.pellets[i].position-worlds.states[0].pellets[i].position)==0,"CPU particle physics differs");
        }
        worlds.states[0].steps=fd::kEpisodeSteps-1;
        fd::CpuBatch terminal=worlds;
        terminal.step(action,nullptr,nullptr,nullptr,false,true);
        float done=0,final_obs[fd::kObs]{};
        worlds.step(action,nullptr,&done,final_obs);
        check(done==1 && worlds.states[0].steps==0,"Time-limit reset failed");
        for(int i=0;i<fd::kObs;++i) check(final_obs[i]==terminal.observations[i],"Truncation bootstrap used reset observation");
        rl::CpuFeedingTrainer learner(16,source,{});
        const auto before=learner.actor.params();const auto started=std::chrono::steady_clock::now();
        learner.update();
        const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
        check(learner.actor.params()!=before && learner.actor_opt.steps()==32,"CPU PPO did not update weights/optimizer");
        cudarl::checkpoint::Writer saved;learner.save(saved);
        rl::CpuFeedingTrainer restored(16,source,{});cudarl::checkpoint::Reader in{saved.data};restored.load(in);in.finish();
        learner.update();restored.update();
        check(learner.actor.params()==restored.actor.params() && learner.critic.params()==restored.critic.params() && learner.log_std==restored.log_std,"Resumed CPU PPO diverges from uninterrupted training");
        check(learner.env.states[0].steps==restored.env.states[0].steps && learner.env.states[0].body.position.x==restored.env.states[0].body.position.x,"CPU worlds did not resume");
        const auto directory=std::filesystem::temp_directory_path()/std::filesystem::path("cudarl-cpu-feeding-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(directory);const auto path=(directory/"training.bin").string();
        auto session=cudarl::demos::create_cpu_feeding_session(argv[1],path,16);
        check(session->drop({0,0,0})>=0,"Live CPU feeding failed");session->set_current({.005f,0,0});
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(90);
        while(session->stats().evaluations<1 && session->stats().error.empty() && std::chrono::steady_clock::now()<deadline) {
            session->step();std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        auto stats=session->stats();
        check(stats.error.empty() && stats.update>=16 && stats.evaluations>0 && stats.learner_changed,"CPU session did not learn and validate");
        check(stats.trained_context==stats.context_version && stats.backend=="CPU","CPU did not practice user context or report backend");
        session->set_paused(true);session->stop();const auto physical=session->state();const int update=session->stats().saved_update;session.reset();
        auto resumed=cudarl::demos::create_cpu_feeding_session(argv[1],path,16);
        check(resumed->state().steps==physical.steps && resumed->state().eaten==physical.eaten && fd::count(resumed->state())==fd::count(physical),"CPU live pond did not resume");
        const auto resume_deadline=std::chrono::steady_clock::now()+std::chrono::seconds(20);
        while(resumed->stats().update<=update && resumed->stats().error.empty() && std::chrono::steady_clock::now()<resume_deadline) std::this_thread::sleep_for(std::chrono::milliseconds(20));
        check(resumed->stats().resumed && resumed->stats().loaded_update==update && resumed->stats().update>update,"CPU full session failed to continue learning");
        resumed->stop();resumed.reset();std::filesystem::remove(directory/"training.bin");std::filesystem::remove(directory/"training.bin.lock");std::filesystem::remove(directory);
        std::printf("CPU_FEEDING checks=%d update_ms=%.1f steps_per_second=%.0f validations=%d adopted=%d PASS\n",checks,seconds*1000,1024/seconds,stats.evaluations,stats.applied_update);return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"CPU_FEEDING FAIL after %d checks: %s\n",checks,e.what());return 1;}
}
