#include <cmath>
#include "check.h"
#include "envs/swimmer/swimmer.h"
#include "envs/swimmer/inertial.h"
#include "gpu_util.h"
#include "rl/swimmer_ppo.h"
namespace sw=cudarl::swimmer;
namespace phys=cudarl::swimmer::inertial;

GPU_TEST_CASE("swimmer evaluation: fixed trials do not alter learning state") {
    cudarl::rl::SwimmerConfig cfg;cfg.num_envs=128;cfg.body=sw::body_preset(8);
    cudarl::rl::SwimmerTrainer reference(cfg);
    cfg.evaluation_every=1;
    cudarl::rl::SwimmerTrainer measured(cfg);
    CHECK(measured.evaluation_update==0 && measured.evaluation_successes==0);
    const auto initial=measured.actor_params();
    measured.evaluate();
    CHECK(measured.evaluation_successes==0 && measured.actor_params()==initial);
    for(int i=0;i<2;++i) {
        reference.update();const auto stats=measured.update();
        CHECK(stats.evaluation_update==i+1 && stats.evaluation_count==64);
        CHECK(stats.evaluation_successes>=0 && stats.evaluation_successes<=64);
        CHECK(reference.actor_params()==measured.actor_params());
    }
}

TEST_CASE("swimmer inertia: an unpowered resting body remains at rest") {
    sw::State s;s.body=sw::body_preset(6);const float action[2]{};
    for(int i=0;i<1000;++i) phys::step(s,action,0.005f,sw::kLimit);
    CHECK(s.x==0 && s.y==0 && s.heading==0 && s.work==0);
}
TEST_CASE("swimmer inertia: vacuum conserves uniform translation") {
    sw::State s;s.body=sw::body_preset(6);s.velocity[0]=2;const float action[2]{};
    const float energy=phys::kinetic_energy(s);
    for(int i=0;i<200;++i) phys::step(s,action,0.005f,sw::kLimit,0,0);
    CHECK(std::fabs(s.x-2.f)<0.001f);
    CHECK(std::fabs(s.velocity[0]-2.f)<0.001f);
    CHECK(std::fabs(phys::kinetic_energy(s)/energy-1.f)<0.001f);
}
TEST_CASE("swimmer inertia: water coasting matches the analytic drag equation") {
    sw::State s;s.body=sw::body_preset(6);s.velocity[0]=2;const float action[2]{};
    const auto full=phys::assemble(s),viscous=phys::assemble(s,0,0.001f);
    const float mass=full.M[0][0],linear=viscous.R[0][0],v0=2*phys::kScale;
    const float quadratic=(full.R[0][0]-linear)/v0;
    const float attenuation=std::exp(-linear/mass);
    const float expected=v0*attenuation/(1.f+quadratic*v0/linear*(1.f-attenuation));
    float energy=phys::kinetic_energy(s);
    phys::step(s,action,0.005f,sw::kLimit);
    CHECK(s.x>0 && s.velocity[0]>0 && s.velocity[0]<2);
    for(int i=1;i<200;++i) {
        phys::step(s,action,0.005f,sw::kLimit);
        const float next=phys::kinetic_energy(s);
        CHECK(next<=energy*1.0001f);energy=next;
    }
    CHECK(std::fabs(s.velocity[0]*phys::kScale/expected-1.f)<0.01f);
}
TEST_CASE("swimmer inertia: joint torques do not supply external linear momentum") {
    const auto drift=[](float dt) {
        sw::State s;s.body=sw::body_preset(6);const float action[2]={0.3f,-0.4f};
        const int count=static_cast<int>(std::round(0.1f/dt));
        for(int i=0;i<count;++i) phys::step(s,action,dt,sw::kLimit,0,0);
        const auto d=phys::assemble(s,0,0);float px=0,py=0,L=0;
        for(int j=0;j<5;++j) {px+=d.M[0][j]*d.v[j];py+=d.M[1][j]*d.v[j];L+=d.M[2][j]*d.v[j];}
        const float c=std::cos(s.heading),n=std::sin(s.heading),world_x=c*px-n*py,world_y=n*px+c*py;
        const float angular=L+s.x*phys::kScale*world_y-s.y*phys::kScale*world_x;
        CHECK(std::fabs(angular)<2e-8f);
        return std::hypot(world_x,world_y);
    };
    const float coarse=drift(0.002f),fine=drift(0.001f);
    CHECK(fine<coarse);
    CHECK(fine<2e-6f);
}
TEST_CASE("swimmer inertia: water resistance is passive and geometry affects it") {
    sw::State s;s.body=sw::body_preset(6);
    s.velocity[0]=1;s.velocity[1]=0.5f;s.velocity[2]=1;s.rate[0]=-1;s.rate[1]=2;
    const auto wide=phys::assemble(s);s.body.fin_normal=2;const auto narrow=phys::assemble(s);
    float dissipation=0;
    for(int i=0;i<5;++i) for(int j=0;j<5;++j) dissipation+=wide.v[i]*wide.R[i][j]*wide.v[j];
    CHECK(dissipation>0);
    CHECK(wide.R[1][1]>narrow.R[1][1]);
    CHECK(wide.M[0][0]>narrow.M[0][0]);
}

GPU_TEST_CASE("swimmer inertia: GPU force integration matches CPU articulated dynamics") {
    constexpr int n=32;const auto body=sw::body_preset(8);
    sw::Batch gpu(n,155,body);std::vector<sw::State> cpu(n),actual(n);
    std::vector<uint32_t> episodes(n,0);std::vector<float> actions(n*2);device_ptr<float> device_actions(n*2);
    for(int i=0;i<n;++i) cpu[i]=sw::initial(155,static_cast<uint32_t>(i),0,body);
    for(int t=0;t<200;++t) {
        for(int i=0;i<n;++i) {
            actions[i*2]=std::sin((t+i)*0.1f);actions[i*2+1]=std::cos((t+i)*0.13f);
            const auto result=sw::advance(cpu[i],actions.data()+i*2);
            if(result.success||result.timeout) cpu[i]=sw::initial(155,static_cast<uint32_t>(i),++episodes[i],body);
        }
        device_actions.copy_from_host(actions.data(),actions.size());gpu.step(device_actions.get(),{},nullptr);
    }
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
    CUDA_CHECK_THROW(cudaMemcpy(actual.data(),gpu.states(),sizeof(sw::State)*n,cudaMemcpyDeviceToHost));
    for(int i=0;i<n;++i) {
        CHECK(std::isfinite(actual[i].x) && std::isfinite(actual[i].y));
        CHECK(actual[i].steps==cpu[i].steps);
        CHECK(std::fabs(actual[i].x-cpu[i].x)<0.003f);
        CHECK(std::fabs(actual[i].y-cpu[i].y)<0.003f);
        CHECK(std::fabs(actual[i].heading-cpu[i].heading)<0.003f);
    }
}
