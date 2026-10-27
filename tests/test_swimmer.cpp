#include <cmath>
#include <vector>
#include "check.h"
#include "envs/swimmer/swimmer.h"
namespace sw=cudarl::swimmer;

TEST_CASE("swimmer: passive bodies cannot create propulsion") {
    const float action[2]{};
    for(int kind=0;kind<=7;++kind) {
        auto state=sw::initial(19,3,0,sw::body_preset(kind));
        const auto initial=state;
        for(int t=0;t<200;++t) sw::advance(state,action);
        CHECK(state.x==initial.x && state.y==initial.y && state.heading==initial.heading);
    }
}

TEST_CASE("swimmer: movement is invariant to world rotation and translation") {
    const float rotation=0.7f,c=std::cos(rotation),s=std::sin(rotation);
    for(int kind=0;kind<=7;++kind) {
        auto a=sw::initial(135,1,0,sw::body_preset(kind)),b=a;
        b.x=3;b.y=-2;b.heading+=rotation;
        b.food_x=3+c*a.food_x-s*a.food_y;b.food_y=-2+s*a.food_x+c*a.food_y;
        for(int t=0;t<100;++t) {
            const float action[2]={std::sin(t*0.17f),std::cos(t*0.13f)};
            sw::advance(a,action);sw::advance(b,action);
        }
        CHECK(std::fabs(b.x-(3+c*a.x-s*a.y))<2e-5f);
        CHECK(std::fabs(b.y-(-2+s*a.x+c*a.y))<2e-5f);
        CHECK(std::fabs(sw::distance(a)-sw::distance(b))<2e-5f);
    }
}

TEST_CASE("swimmer: a reciprocal one-joint stroke has negligible net travel") {
    for(int kind=0;kind<=7;++kind) {
        sw::State state;state.body=sw::body_preset(kind);
        for(int t=0;t<5;++t) {const float action[2]={0.2f,0};sw::integrate(state,action);}
        for(int t=0;t<5;++t) {const float action[2]={-0.2f,0};sw::integrate(state,action);}
        CHECK(std::hypot(state.x,state.y)<0.002f);
    }
}

TEST_CASE("swimmer: success and time limit have distinct bootstrap semantics") {
    sw::State success;success.food_x=0.1f;
    const float action[2]{};
    const auto reached=sw::advance(success,action);
    CHECK(reached.success && !reached.timeout && reached.reward>2.f);
    auto timeout=sw::initial(33,2);timeout.steps=sw::kMaxSteps-1;
    const auto ended=sw::advance(timeout,action);
    CHECK(!ended.success && ended.timeout);
}

TEST_CASE("swimmer: a wider membrane changes forces and motion, not just rendering") {
    auto narrow=sw::initial(5,1),wide=narrow;
    wide.body.fin_normal=4.f;
    for(int t=0;t<200;++t) {
        const float action[2]={std::cos(t*0.2f),std::sin(t*0.2f)};
        sw::advance(narrow,action);sw::advance(wide,action);
    }
    CHECK(std::hypot(narrow.x-wide.x,narrow.y-wide.y)>0.01f);
    CHECK(std::fabs(narrow.work-wide.work)>0.01f);
}

GPU_TEST_CASE("swimmer: all bodies match CPU physics and preserve terminal observations") {
    constexpr int n=32;
    for(int kind=0;kind<=7;++kind) {
        const auto body=sw::body_preset(kind);
        sw::Batch gpu(n,7321,body);
        std::vector<sw::State> cpu(n),actual(n);
        std::vector<uint32_t> episodes(n,0);
        std::vector<float> tape(n*2),host_reward(n),host_done(n),host_truncated(n),host_final(n*sw::kObs);
        device_ptr<float> actions(n*2),reward(n),done(n),truncated(n),final_obs(n*sw::kObs);
        for(int i=0;i<n;++i) cpu[i]=sw::initial(7321,static_cast<uint32_t>(i),0,body);
        for(int t=0;t<sw::kMaxSteps+2;++t) {
            std::vector<sw::Result> results(n);std::vector<float> finals(n*sw::kObs);
            for(int i=0;i<n;++i) {
                tape[i*2]=std::sin((t+i)*0.17f);tape[i*2+1]=std::cos((t+i)*0.11f);
                results[i]=sw::advance(cpu[i],tape.data()+i*2);
                if(results[i].success||results[i].timeout) {
                    sw::observe(cpu[i],finals.data()+i*sw::kObs);
                    cpu[i]=sw::initial(7321,static_cast<uint32_t>(i),++episodes[i],body);
                }
            }
            actions.copy_from_host(tape.data(),tape.size());
            gpu.step(actions.get(),{reward.get(),done.get(),truncated.get(),final_obs.get()},nullptr);
            if(t%100!=0 && t!=sw::kMaxSteps-1) continue;
            CUDA_CHECK_THROW(cudaDeviceSynchronize());
            CUDA_CHECK_THROW(cudaMemcpy(actual.data(),gpu.states(),sizeof(sw::State)*n,cudaMemcpyDeviceToHost));
            reward.copy_to_host(host_reward.data(),n);done.copy_to_host(host_done.data(),n);
            truncated.copy_to_host(host_truncated.data(),n);final_obs.copy_to_host(host_final.data(),host_final.size());
            for(int i=0;i<n;++i) {
                CHECK(std::fabs(actual[i].x-cpu[i].x)<0.002f);
                CHECK(std::fabs(actual[i].y-cpu[i].y)<0.002f);
                CHECK(std::isfinite(actual[i].work) && actual[i].work>=0);
                CHECK(std::fabs(actual[i].work-cpu[i].work)<0.01f);
                CHECK(actual[i].steps==cpu[i].steps);
                CHECK(std::fabs(actual[i].body.motor_speed-body.motor_speed)<1e-6f);
                CHECK(std::fabs(host_reward[i]-results[i].reward)<0.001f);
                CHECK((host_done[i]!=0)==(results[i].success||results[i].timeout));
                CHECK((host_truncated[i]!=0)==results[i].timeout);
                if(host_done[i]!=0) for(int j=0;j<sw::kObs;++j)
                    CHECK(std::fabs(host_final[i*sw::kObs+j]-finals[i*sw::kObs+j])<0.002f);
            }
        }
    }
}
