#include "rl/shelter_motor_run.h"
#include "rl/shelter_calibration.h"
#include "rl/shelter_hold_cases.h"
#include <atomic>
#include <csignal>
#include <cstdio>
#include <filesystem>

namespace {std::atomic<bool> stopping{false};void stop(int) {stopping.store(true);}}
int main(int argc,char** argv) {
    using namespace cudarl;
    try {
        if(argc<4 || argc>5) throw std::invalid_argument("Usage: shelter_motor_train <full-boulder-save> <output-full-motor-snapshot> <total-updates> [worlds=1024]");
        const auto source=rl::load_boulder_source(argv[1]);const std::string path=argv[2];
        if(std::filesystem::weakly_canonical(argv[1])==std::filesystem::weakly_canonical(path)) throw std::invalid_argument("Motor practice must never overwrite its boulder source");
        const int total=std::stoi(argv[3]),worlds=argc==5?std::stoi(argv[4]):1024;
        if(total<1 || total>1000000 || worlds<16 || worlds>2048 || worlds%8) throw std::invalid_argument("Invalid motor practice budget");
        constexpr uint32_t seed=73;auto context=source.context;context.current={.0075f,0,0};
        rl::ShelterActionCalibration calibration;
        const auto calibrated=rl::calibrate_shelter_actions(source,&calibration);
        const auto legacy_actor=rl::transfer_shelter_motor(source.motor);
        const auto initial_actor=rl::normalized_shelter_actor(legacy_actor,calibrated);
        rl::ShelterMotorTrainer trainer(worlds,seed);trainer.actor.set_params(initial_actor);
        trainer.actor_opt.set_learning_rate(.0001f);trainer.critic_opt.set_learning_rate(.0003f);trainer.std_opt.set_learning_rate(.0001f);
        const std::vector<float> deviations(4,-1.5f);trainer.log_std.copy_from_host(deviations.data(),deviations.size());
        trainer.env.configure(source.ancestry.policy.actor,context,trainer.stream.value,true);
        trainer.env.set_action_scales(calibrated);
        if(checkpoint::exists(path)) {
            const auto saved=rl::load_shelter_motor_run(path);
            if(saved.source.parent!=source.parent || saved.source.fingerprint!=source.fingerprint || saved.worlds!=worlds || saved.seed!=seed)
                throw std::runtime_error("Motor resume source or protocol mismatch; both saves preserved");
            rl::restore_shelter_motor(trainer,saved.training);
            for(int j=0;j<4;++j) if(saved.scales.value[j]!=calibrated.value[j]) throw std::runtime_error("Action calibration protocol changed");
        }
        const auto inherited=initial_actor;const auto start=trainer.actor.get_params();
        for(int j=0;j<4;++j) std::printf("SHELTER_CALIBRATION joint=%d scale=%.9g median=%.9g p95=%.9g maximum=%.9g static_max=%.9g samples=%d fingerprint=%llu\n",
            j,calibrated.value[j],calibration.median[j],calibration.p95[j],calibration.maximum[j],calibration.static_maximum[j],calibration.samples,
            static_cast<unsigned long long>(calibration.fingerprint));
        const auto baseline_probe=rl::probe_shelter_motor(inherited,rl::shelter_motor_probe_cases(source.frame,0x571EF101u),240,100,calibrated);
        int baseline_holds=0,baseline_arrivals=0;
        for(const auto& result:baseline_probe) {baseline_holds+=result.longest_safe_dwell_ticks>=100;baseline_arrivals+=result.arrival_step>=0;}
        std::printf("SHELTER_CALIBRATED_FIXED_BASELINE arrivals=%d holds=%d/%zu trained=0\n",baseline_arrivals,baseline_holds,baseline_probe.size());
        const auto hold_cases=rl::shelter_hold_cases(source.frame);
        const auto initial_holding=rl::probe_shelter_motor(inherited,hold_cases,1,200,calibrated);
        int initial_holds=0;float initial_longest=0;
        for(const auto& result:initial_holding) {initial_holds+=result.longest_safe_dwell_ticks>=100;initial_longest=std::max(initial_longest,result.longest_safe_dwell_ticks*deep::dt);}
        std::printf("SHELTER_CALM_HOLD_BASELINE five_second_holds=%d/%zu longest_s=%.2f trained=0 assigned_task=1\n",initial_holds,initial_holding.size(),initial_longest);
        std::printf("SHELTER_MOTOR_START parent=%llu motor_update=%d learner_update=%d resumed_update=%d worlds=%d requested_total=%d inputs=59 native_ppo=1\n",
            static_cast<unsigned long long>(source.fingerprint),source.motor_update,source.learner_update,trainer.update_id,worlds,total);std::fflush(stdout);
        std::signal(SIGINT,stop);std::signal(SIGTERM,stop);auto last_save=rl::Clock::now();
        while(trainer.update_id<total && !stopping.load()) {
            const double seconds=trainer.update_seconds();
            // 25*64 is an exact multiple of the 400-step cover episode. Read
            // the peak before that reset instead of reporting fresh zeros.
            if(trainer.update_id==1 || trainer.update_id%25==24 || trainer.update_id==total) {
                const auto states=trainer.env.host_states(trainer.stream.value);int cover=0,food=0,max_hold=0,held=0,invalid=0;
                double exposure=0;
                for(const auto& state:states) {
                    invalid+=state.state.pond.invalid?1:0;
                    if(state.request.cover) {++cover;exposure+=state.exposure;max_hold=std::max(max_hold,state.peak_hold);held+=state.peak_hold>=100?1:0;}
                    else food+=state.state.pond.eaten;
                }
                std::printf("SHELTER_MOTOR_TRAIN update=%d seconds=%.4f cover_worlds=%d practice_full_five_second_holds=%d longest_hold_s=%.2f mean_exposure=%.4f food_world_eaten=%d invalid=%d\n",
                    trainer.update_id,seconds,cover,held,max_hold*deep::dt,cover?exposure/cover:0,food,invalid);std::fflush(stdout);
            }
            if(trainer.update_id==1 || trainer.update_id%25==0 || rl::Clock::now()-last_save>=std::chrono::seconds(30)) {
                rl::save_shelter_motor_run(path,source,trainer);last_save=rl::Clock::now();
            }
        }
        rl::save_shelter_motor_run(path,source,trainer);const auto learned=trainer.actor.get_params();int changed=0,changed_run=0;
        float learned_std[4];trainer.log_std.copy_to_host(learned_std,4);
        std::printf("SHELTER_EXPLORATION log_std=(%.6f,%.6f,%.6f,%.6f) sigma=(%.6f,%.6f,%.6f,%.6f)\n",learned_std[0],learned_std[1],learned_std[2],learned_std[3],expf(learned_std[0]),expf(learned_std[1]),expf(learned_std[2]),expf(learned_std[3]));
        for(size_t i=0;i<learned.size();++i) {changed+=learned[i]!=inherited[i];changed_run+=learned[i]!=start[i];}
        const uint32_t comparison_seed=0x571EF001u,holdout_seed=0x571EF101u;
        const auto probe=rl::probe_shelter_motor(learned,rl::shelter_motor_probe_cases(source.frame,holdout_seed),240,100,trainer.env.action_scales());
        int arrived=0,held=0,finite=0;double exposure=0;
        for(size_t i=0;i<probe.size();++i) {
            const auto& r=probe[i];arrived+=r.arrival_step>=0;held+=r.longest_safe_dwell_ticks>=100;finite+=r.finite?1:0;exposure+=r.mean_exposure;
            std::printf("SHELTER_MOTOR_HOLDOUT case=%zu arrived=%d full_five_second_hold=%d longest_hold_s=%.2f mean_exposure=%.4f final_goal_distance=%.4f clearance=%.4f finite=%d\n",
                i,r.arrival_step>=0?1:0,r.longest_safe_dwell_ticks>=100?1:0,r.longest_safe_dwell_ticks*deep::dt,r.mean_exposure,r.final_goal_distance,r.minimum_clearance,r.finite?1:0);
        }
        const double base_calm=rl::shelter_motor_food_rate(legacy_actor,source,comparison_seed),calm=rl::shelter_motor_food_rate(learned,source,comparison_seed,16,600,{},trainer.env.action_scales());
        const double base_flow=rl::shelter_motor_food_rate(legacy_actor,source,comparison_seed,16,600,{.0075f,0,0});
        const double flow=rl::shelter_motor_food_rate(learned,source,comparison_seed,16,600,{.0075f,0,0},trainer.env.action_scales());
        const auto holding=rl::probe_shelter_motor(learned,hold_cases,1,200,trainer.env.action_scales());
        int holding_success=0;double longest_total=0;
        for(size_t i=0;i<holding.size();++i) {
            const auto& r=holding[i];holding_success+=r.longest_safe_dwell_ticks>=100;longest_total+=r.longest_safe_dwell_ticks*deep::dt;
            std::printf("SHELTER_CALM_HOLD case=%zu five_second_hold=%d longest_s=%.2f mean_exposure=%.4f clearance=%.4f finite=%d\n",i,r.longest_safe_dwell_ticks>=100?1:0,r.longest_safe_dwell_ticks*deep::dt,r.mean_exposure,r.minimum_clearance,r.finite?1:0);
        }
        std::printf("SHELTER_CALM_HOLD_DONE initial=%d/%zu trained=%d/%zu mean_longest_s=%.3f assigned_task=1 escape_claim=0\n",initial_holds,initial_holding.size(),holding_success,holding.size(),longest_total/holding.size());
        std::printf("SHELTER_MOTOR_DONE update=%d changed_from_parent=%d changed_this_run=%d holdout_cases=%zu arrivals=%d full_five_second_holds=%d finite=%d mean_exposure=%.4f inherited_calm_food_min=%.3f current_calm_food_min=%.3f inherited_flow_food_min=%.3f current_flow_food_min=%.3f deployment_approved=0 saved=%s\n",
            trainer.update_id,changed,changed_run,probe.size(),arrived,held,finite,exposure/probe.size(),base_calm,calm,base_flow,flow,path.c_str());std::fflush(stdout);
        return finite==static_cast<int>(probe.size())?0:1;
    } catch(const std::exception& error) {std::fprintf(stderr,"SHELTER_MOTOR_ERROR %s\n",error.what());return 1;}
}
