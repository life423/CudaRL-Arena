#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include "check.h"
#include "envs/swimmer/foraging.h"
#include "rl/swimmer_ppo.h"
#include "nn/mlp_cpu.h"
namespace fg=cudarl::foraging;
namespace sw=cudarl::swimmer;

TEST_CASE("foraging: initial and replacement food cover the pond without clumping") {
    int outside_old_area=0,total=0,quadrants[4]{};
    float min_x=100,max_x=-100,min_y=100,max_y=-100;
    float smallest_gap=100;
    for(uint32_t seed=1;seed<=500;++seed) {
        auto state=fg::initial(seed,9);
        for(int round=0;round<4;++round) {
            for(int i=0;i<3;++i) {
                const float x=state.food_x[i],y=state.food_y[i];
                min_x=std::min(min_x,x);max_x=std::max(max_x,x);
                min_y=std::min(min_y,y);max_y=std::max(max_y,y);
                if(x*x/6.25f+y*y/3.24f>1.f) ++outside_old_area;
                ++quadrants[(x>=0?1:0)+(y>=0?2:0)];++total;
                CHECK(x*x/(4.6f*4.6f)+y*y/(3.5f*3.5f)<0.75f);
                for(int j=0;j<i;++j) smallest_gap=std::min(smallest_gap,std::hypot(x-state.food_x[j],y-state.food_y[j]));
            }
            // Replacement must remain global even when the animal is off-center.
            state.body.x=seed%2==0?3.f:-3.f;state.body.y=0;
            CHECK(fg::place_food(state,round%3));
        }
    }
    CHECK(smallest_gap>=2.2f-1e-5f);
    CHECK(min_x< -3.7f && max_x>3.7f && min_y< -2.6f && max_y>2.6f);
    CHECK(outside_old_area>total*0.65);
    for(int count:quadrants) CHECK(count>total*0.15);
    std::printf("  FOOD_COVERAGE outside_old=%.1f%% gap=%.3f x=[%.2f,%.2f] y=[%.2f,%.2f]\n",
        100.0*outside_old_area/total,smallest_gap,min_x,max_x,min_y,max_y);
}

TEST_CASE("foraging: all three stable food slots reach the policy") {
    auto state=fg::initial(42,3);state.body.x=0;state.body.y=0;state.body.heading=0;
    state.food_x[0]=1;state.food_y[0]=0;
    state.food_x[1]=-1;state.food_y[1]=1;
    state.food_x[2]=0.2f;state.food_y[2]=-0.4f;
    float obs[fg::kObs];fg::observe(state,obs);
    CHECK(obs[0]==0.5f && obs[1]==0 && obs[2]==-0.5f && obs[3]==0.5f);
    CHECK(obs[4]==0.1f && obs[5]==-0.2f);
    std::swap(state.food_x[0],state.food_x[2]);std::swap(state.food_y[0],state.food_y[2]);
    fg::observe(state,obs);
    CHECK(obs[0]==0.1f && obs[4]==0.5f); // No nearest-food sorting or target selection.
}
TEST_CASE("foraging: replacements leave the eaten region and cannot be farmed in place") {
    float shortest=100;int failures=0;
    for(uint32_t seed=1;seed<=1000;++seed) {
        auto state=fg::initial(seed,3);
        for(int round=0;round<24;++round) {
            const int slot=round%3;const auto before=state;
            state.body.x=state.food_x[slot];state.body.y=state.food_y[slot];
            if(!fg::place_food(state,slot)) {++failures;continue;}
            const float distance=std::hypot(state.food_x[slot]-before.food_x[slot],state.food_y[slot]-before.food_y[slot]);
            shortest=std::min(shortest,distance);
            CHECK(fg::region(state.food_x[slot],state.food_y[slot])!=fg::region(before.food_x[slot],before.food_y[slot]));
            for(int j=0;j<3;++j) if(j!=slot) CHECK(state.food_x[j]==before.food_x[j] && state.food_y[j]==before.food_y[j]);
        }
    }
    CHECK(failures==0 && shortest>=2.6f-1e-5f);
    std::printf("  FOOD_RELOCATION trials=24000 failed=%d shortest=%.3f\n",failures,shortest);
}
TEST_CASE("foraging: holding the tail against a stop is worse than releasing it") {
    auto outward=fg::initial(19,1),inward=outward;
    outward.body.angle[0]=inward.body.angle[0]=1.25f;
    outward.body.angle[1]=inward.body.angle[1]=1.25f;
    const float push[2]={0.5f,0.5f},release[2]={-0.5f,-0.5f};
    const auto blocked=fg::advance(outward,push),free=fg::advance(inward,release);
    CHECK(free.reward>blocked.reward);
    CHECK(std::fabs(inward.body.angle[0])+std::fabs(inward.body.angle[1])<std::fabs(outward.body.angle[0])+std::fabs(outward.body.angle[1]));
}
TEST_CASE("foraging: learned recovery releases folded tails without scripted movement") {
    const auto path=std::filesystem::path(__FILE__).parent_path()/"fixtures"/"foraging-recovery-policy.json";
    std::ifstream file(path);
    CHECK(file.good());if(!file) return;
    const std::string json((std::istreambuf_iterator<char>(file)),std::istreambuf_iterator<char>());
    const auto start=json.find('[',json.find("\"parameters\"")),end=json.find(']',start);
    CHECK(start!=std::string::npos && end!=std::string::npos);if(start==std::string::npos || end==std::string::npos) return;
    auto values=json.substr(start+1,end-start-1);std::replace(values.begin(),values.end(),',',' ');
    std::istringstream input(values);
    std::vector<float> weights((std::istream_iterator<float>(input)),std::istream_iterator<float>());
    cudarl::nn::CpuMlp<float> controller(cudarl::rl::swimmer_specification(2,13));
    CHECK(weights.size()==controller.params().size());if(weights.size()!=controller.params().size()) return;
    controller.params()=weights;
    int recovered=0;
    for(uint32_t trial=0;trial<64;++trial) {
        auto state=fg::initial(0x71AB93u,trial);
        state.body.angle[0]=(trial&1u)?1.25f:-1.25f;
        state.body.angle[1]=(trial&2u)?1.25f:-1.25f;
        float distance=0;bool released=false;
        for(int step=0;step<240;++step) {
            float observation[13];fg::observe(state,observation);controller.forward(observation,1);
            fg::advance(state,controller.output());
            distance=std::max(distance,std::hypot(state.body.x,state.body.y));
            released=released || std::fabs(state.body.angle[0])<0.9f || std::fabs(state.body.angle[1])<0.9f;
        }
        if(released && distance>=0.5f) ++recovered;
    }
    CHECK(recovered==64);
}
TEST_CASE("foraging: food spawns clear and only the eaten piece moves") {
    for(uint32_t seed=1;seed<=200;++seed) {
        auto state=fg::initial(seed,9);
        for(int i=0;i<fg::kFoods;++i) CHECK(fg::valid_food(state,i,state.food_x[i],state.food_y[i]));
        const int slot=static_cast<int>(seed%3);const auto before=state;
        state.food_x[slot]=state.body.x;state.food_y[slot]=state.body.y;
        const float zero[2]{};const auto result=fg::advance(state,zero);
        CHECK(result.eaten_slot==slot && state.eaten==1 && state.since_food_steps==0);
        CHECK(state.body.x==before.body.x && state.body.y==before.body.y);
        for(int i=0;i<fg::kFoods;++i) {
            if(i!=slot) CHECK(state.food_x[i]==before.food_x[i] && state.food_y[i]==before.food_y[i]);
            CHECK(fg::valid_food(state,i,state.food_x[i],state.food_y[i]));
        }
        CHECK(result.reward>2.9f && !result.timeout);
    }
}
TEST_CASE("foraging: unchanged physics and no idle or respawn progress reward") {
    auto state=fg::initial(51,8);auto physics=state.body;
    const float zero[2]{};
    const auto idle=fg::advance(state,zero);
    CHECK(idle.reward<0 && state.body.x==physics.x && state.body.y==physics.y);
    for(int t=0;t<200;++t) {
        const float action[2]={std::sin(t*0.19f),std::cos(t*0.11f)};
        sw::advance_physics(physics,action);fg::advance(state,action);
    }
    CHECK(std::fabs(state.body.x-physics.x)<1e-6f && std::fabs(state.body.y-physics.y)<1e-6f);
}
GPU_TEST_CASE("foraging: CPU and CUDA match across collections and time limits") {
    constexpr int n=32;fg::Batch gpu(n,7321);std::vector<fg::State> cpu(n),actual(n);
    std::vector<uint32_t> episodes(n,0);std::vector<float> tape(n*2);
    device_ptr<float> actions(n*2),reward(n),done(n),truncated(n),final_obs(n*fg::kObs);
    std::vector<float> host_reward(n),host_done(n),host_truncated(n),host_final(n*fg::kObs);
    for(int i=0;i<n;++i) cpu[i]=fg::initial(7321,static_cast<uint32_t>(i));
    for(int t=0;t<=fg::kEpisodeSteps;++t) {
        std::vector<fg::Result> results(n);std::vector<float> finals(n*fg::kObs);
        for(int i=0;i<n;++i) {
            tape[i*2]=std::sin((t+i)*0.17f);tape[i*2+1]=std::cos((t+i)*0.11f);
            results[i]=fg::advance(cpu[i],tape.data()+i*2);
            if(results[i].timeout) {
                fg::observe(cpu[i],finals.data()+i*fg::kObs);
                cpu[i]=fg::initial(7321,static_cast<uint32_t>(i),++episodes[i]);
            }
        }
        actions.copy_from_host(tape.data(),tape.size());
        gpu.step(actions.get(),{reward.get(),done.get(),truncated.get(),final_obs.get()},nullptr);
        if(t%200!=0 && t!=fg::kEpisodeSteps-1) continue;
        CUDA_CHECK_THROW(cudaDeviceSynchronize());
        CUDA_CHECK_THROW(cudaMemcpy(actual.data(),gpu.states(),sizeof(fg::State)*n,cudaMemcpyDeviceToHost));
        reward.copy_to_host(host_reward.data(),n);done.copy_to_host(host_done.data(),n);
        truncated.copy_to_host(host_truncated.data(),n);final_obs.copy_to_host(host_final.data(),host_final.size());
        for(int i=0;i<n;++i) {
            CHECK(std::fabs(actual[i].body.x-cpu[i].body.x)<0.003f);
            CHECK(std::fabs(actual[i].body.y-cpu[i].body.y)<0.003f);
            CHECK(actual[i].eaten==cpu[i].eaten && actual[i].body.steps==cpu[i].body.steps);
            CHECK(std::fabs(host_reward[i]-results[i].reward)<0.002f);
            CHECK((host_done[i]!=0)==results[i].timeout && (host_truncated[i]!=0)==results[i].timeout);
            for(int j=0;j<fg::kFoods;++j) {
                CHECK(std::fabs(actual[i].food_x[j]-cpu[i].food_x[j])<0.003f);
                CHECK(std::fabs(actual[i].food_y[j]-cpu[i].food_y[j])<0.003f);
            }
            if(host_done[i]!=0) for(int j=0;j<fg::kObs;++j) CHECK(std::fabs(host_final[i*fg::kObs+j]-finals[i*fg::kObs+j])<0.003f);
        }
    }
}
GPU_TEST_CASE("foraging: evaluation preserves the 13-input learner and optimizer") {
    cudarl::rl::SwimmerConfig cfg;cfg.num_envs=128;cfg.body=sw::body_preset(8);
    cudarl::rl::ForagingTrainer reference(cfg);cfg.evaluation_every=1;
    cudarl::rl::ForagingTrainer measured(cfg);
    CHECK(measured.actor_params().size()==1570 && measured.evaluation_foods==0);
    for(int i=0;i<2;++i) {
        reference.update();const auto stats=measured.update();
        CHECK(stats.evaluation_count==64 && std::fabs(stats.evaluation_seconds-60)<0.001);
        CHECK(reference.actor_params()==measured.actor_params());
    }
}
