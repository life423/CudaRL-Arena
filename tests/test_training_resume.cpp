#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>
#include "check.h"
#include "demos/ppo_session.h"
#include "rl/swimmer_ppo.h"
namespace {
namespace cp=cudarl::checkpoint;
namespace rl=cudarl::rl;
std::string temporary(const char* tag) {
    const auto directory=std::filesystem::path(__FILE__).parent_path().parent_path()/"build"/"pond-learning-checks";
    std::filesystem::create_directories(directory);
    return (directory/(std::string(tag)+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".training")).u8string();
}
template<class F> bool wait_for_resume(F condition) {
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(20);
    while(!condition() && std::chrono::steady_clock::now()<deadline) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    return condition();
}
void compare(const rl::ForagingTrainer& a,const rl::ForagingTrainer& b) {
    CHECK(a.update_id==b.update_id && a.actor_params()==b.actor_params() && a.critic.get_params()==b.critic.get_params());
    float x[2],y[2];a.log_std.copy_to_host(x,2);b.log_std.copy_to_host(y,2);CHECK(x[0]==y[0] && x[1]==y[1]);
    cp::Writer sa,sb;a.actor_opt.save(sa);b.actor_opt.save(sb);a.critic_opt.save(sa);b.critic_opt.save(sb);a.std_opt.save(sa);b.std_opt.save(sb);
    CHECK(sa.data==sb.data);
    std::vector<float> oa(static_cast<size_t>(a.n)*13),ob(oa.size());
    CUDA_CHECK_THROW(cudaMemcpy(oa.data(),a.env.obs(),oa.size()*sizeof(float),cudaMemcpyDeviceToHost));
    CUDA_CHECK_THROW(cudaMemcpy(ob.data(),b.env.obs(),ob.size()*sizeof(float),cudaMemcpyDeviceToHost));CHECK(oa==ob);
    CHECK(a.best_policy().parameters==b.best_policy().parameters && a.best_policy().foods==b.best_policy().foods);
}
}
GPU_TEST_CASE("training resume: exact PPO continuation across timeouts and evaluation for both input modes") {
    for(bool shuffled:{false,true}) {
        rl::SwimmerConfig cfg;cfg.num_envs=128;cfg.evaluation_every=25;cfg.body=cudarl::swimmer::body_preset(8);cfg.shuffle_food_inputs=shuffled;
        rl::ForagingTrainer uninterrupted(cfg);
        for(int i=0;i<18;++i) uninterrupted.update(); // 1,152 of 1,200 control steps.
        const auto path=temporary(shuffled?"shuffled-resume-":"canonical-resume-");
        cp::write_atomic(path,uninterrupted.training_state());
        cfg.training_checkpoint_path=path;
        {
            rl::ForagingTrainer resumed(cfg);CHECK(resumed.checkpoint_status().resumed && resumed.checkpoint_status().loaded_update==18);
            compare(uninterrupted,resumed);
            for(int i=0;i<8;++i) {uninterrupted.update();resumed.update();compare(uninterrupted,resumed);}
            CHECK(resumed.update_id==26 && resumed.evaluation_update==25);
            resumed.save_checkpoint();CHECK(resumed.checkpoint_status().saved_update==26);
            bool busy=false;try {rl::ForagingTrainer duplicate(cfg);} catch(const cp::Busy&) {busy=true;}CHECK(busy);
        }
        rl::ForagingTrainer reopened(cfg);compare(uninterrupted,reopened);
        std::filesystem::remove(std::filesystem::u8path(path));
    }
}
TEST_CASE("training resume: checksum rejects corruption and failed replacement preserves the prior save") {
    const auto path=temporary("checkpoint-io-");const std::vector<uint8_t> first{1,2,3,4,5};cp::write_atomic(path,first);CHECK(cp::read(path)==first);
    std::filesystem::create_directory(std::filesystem::u8path(path+".tmp"));
    bool failed=false;try {cp::write_atomic(path,{9,8,7});} catch(const std::exception&) {failed=true;}CHECK(failed && cp::read(path)==first);
    std::filesystem::remove(std::filesystem::u8path(path+".tmp"));
    {std::fstream file(std::filesystem::u8path(path),std::ios::binary|std::ios::in|std::ios::out);file.seekp(24);const char bad=99;file.write(&bad,1);}
    bool corrupt=false;try {cp::read(path);} catch(const std::exception&) {corrupt=true;}CHECK(corrupt);
    std::filesystem::remove(std::filesystem::u8path(path));
}
GPU_TEST_CASE("training resume: session saves on close and publishes the restored update without retraining") {
    rl::SwimmerConfig cfg;cfg.num_envs=128;cfg.body=cudarl::swimmer::body_preset(8);cfg.shuffle_food_inputs=true;cfg.evaluation_every=25;
    cfg.total_updates=0;cfg.training_checkpoint_path=temporary("session-resume-");
    int saved_update=0;std::vector<float> saved_actor;
    {
        cudarl::demos::TrainingSession<rl::ForagingTrainer> session(cfg,0,true);
        CHECK(wait_for_resume([&] {return session.ready() || !session.error().empty();}));CHECK(session.error().empty());
        session.set_paused(false);
        CHECK(wait_for_resume([&] {std::vector<rl::SwimmerStats> h;session.history_since(0,h);return !h.empty() && h.back().update>=3;}));
        session.set_paused(true);std::this_thread::sleep_for(std::chrono::milliseconds(250));
        session.request_save();CHECK(wait_for_resume([&] {return session.checkpoint_status().saved_update>=3;}));
        session.stop();uint64_t version=0;CHECK(session.latest_policy(version,saved_actor,&saved_update));
        CHECK(session.checkpoint_status().saved_update==saved_update);
    }
    {
        cudarl::demos::TrainingSession<rl::ForagingTrainer> session(cfg,0,true);
        CHECK(wait_for_resume([&] {return session.ready() || !session.error().empty();}));CHECK(session.error().empty());
        CHECK(session.checkpoint_status().resumed);uint64_t version=0;int restored_update=-1;std::vector<float> restored;
        CHECK(session.latest_policy(version,restored,&restored_update) && restored==saved_actor && restored_update==saved_update);
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        std::vector<rl::SwimmerStats> h;session.history_since(0,h);CHECK(h.back().update==saved_update);session.stop();
    }
    std::filesystem::remove(std::filesystem::u8path(cfg.training_checkpoint_path));
}
GPU_TEST_CASE("training resume: incompatible saves stay intact and one-time actor import keeps learned weights") {
    rl::SwimmerConfig cfg;cfg.num_envs=128;cfg.body=cudarl::swimmer::body_preset(8);cfg.shuffle_food_inputs=true;cfg.evaluation_every=25;
    cfg.training_checkpoint_path=temporary("migration-");
    cfg.import_policy={350,0,64,1200,0x51A7E123u,1,rl::ForagingTrainer::initial_actor_params(1),true};
    std::vector<float> imported_after;
    {
        rl::ForagingTrainer imported(cfg);CHECK(imported.checkpoint_status().imported && !imported.checkpoint_status().resumed);
        CHECK(imported.update_id==350 && imported.actor_params()==cfg.import_policy.parameters && imported.actor_opt.steps()==0);
        imported.update();CHECK(imported.update_id==351 && imported.actor_opt.steps()==32);imported.save_checkpoint();imported_after=imported.actor_params();
    }
    const auto original_bytes=cp::read(cfg.training_checkpoint_path);
    {
        auto other=cfg;other.shuffle_food_inputs=false;bool rejected=false;
        try {rl::ForagingTrainer incompatible(other);} catch(const std::exception&) {rejected=true;}CHECK(rejected && cp::read(cfg.training_checkpoint_path)==original_bytes);
    }
    {
        rl::ForagingTrainer restored(cfg);CHECK(restored.actor_params()==imported_after && restored.actor_opt.steps()==32 && restored.update_id==351);
    }
    std::filesystem::remove(std::filesystem::u8path(cfg.training_checkpoint_path));
}
