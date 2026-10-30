#include "check.h"
#include "rl/shelter_calibration.h"
#include "rl/gaussian_math.h"
#include <limits>

using namespace cudarl;
namespace {
rl::BoulderSource constant_calibration_source() {
    rl::BoulderSource source;source.frame=arch::initial(7,0,0,false,{2});source.context.basin.layout=2;source.fingerprint=0xC41B2026u;
    const auto spec=rl::swimmer_specification(4,arch::kObs);source.motor.assign(static_cast<size_t>(spec.num_params()),0.f);
    const int bias=spec.bias_offset(spec.num_layers()-1);source.motor[static_cast<size_t>(bias)]=.4f;
    source.motor[static_cast<size_t>(bias+1)]=-.1f;source.motor[static_cast<size_t>(bias+2)]=100.f;source.motor[static_cast<size_t>(bias+3)]=-.5f;
    return source;
}
}
TEST_CASE("shelter action calibration uses legal development poses and preserves source") {
    auto source=constant_calibration_source();const auto original=source.motor;const auto position=source.frame.pond.body.position;
    rl::ShelterActionCalibration report;const auto scale=rl::calibrate_shelter_actions(source,&report);
    CHECK(report.samples==4*(rl::shelter_calibration_per_cover+1));CHECK(report.protocol==rl::shelter_calibration_protocol_fingerprint());
    CHECK(scale.value[0]==1.f);CHECK(scale.value[1]==1.f);CHECK(scale.value[2]==128.f);CHECK(scale.value[3]==1.f);
    CHECK(report.median[2]==100.f);CHECK(report.p95[2]==100.f);CHECK(report.maximum[2]==100.f);CHECK(report.static_maximum[2]==100.f);
    CHECK(source.motor==original);CHECK(source.frame.pond.body.position.x==position.x);
    CHECK(source.frame.pond.body.position.y==position.y);CHECK(source.frame.pond.body.position.z==position.z);
    CHECK(report.fingerprint!=0);CHECK(report.observations!=0);
}
TEST_CASE("shelter action calibration is deterministic and bound to source provenance") {
    auto source=constant_calibration_source();rl::ShelterActionCalibration a,b;
    (void)rl::calibrate_shelter_actions(source,&a);(void)rl::calibrate_shelter_actions(source,&b);
    CHECK(a.fingerprint==b.fingerprint);CHECK(a.observations==b.observations);CHECK(a.samples==b.samples);
    for(int j=0;j<4;++j) {CHECK(a.scale.value[j]==b.scale.value[j]);CHECK(a.p95[j]==b.p95[j]);}
    ++source.fingerprint;(void)rl::calibrate_shelter_actions(source,&b);CHECK(a.fingerprint!=b.fingerprint);
}
TEST_CASE("shelter action calibration rejects nonfinite source parameters") {
    auto source=constant_calibration_source();source.motor[0]=std::numeric_limits<float>::quiet_NaN();
    bool rejected=false;try {(void)rl::calibrate_shelter_actions(source);} catch(const std::exception&) {rejected=true;}CHECK(rejected);
}
GPU_TEST_CASE("shelter action normalization preserves food sampling and restores physical exploration") {
    constexpr int n=256;std::vector<float> raw(n*4),legacy(n*4),input(n*shelter_motor::kObs,0),gradient(n*4,1.f);
    shelter_motor::ActuatorScale scale;scale.value[0]=2;scale.value[1]=1;scale.value[2]=128;scale.value[3]=4;
    const float command[4]={.4f,-.1f,100.f,-.5f};
    for(int i=0;i<n;++i) {input[static_cast<size_t>(i)*shelter_motor::kObs+55]=i%2?1.f:0.f;for(int j=0;j<4;++j) {legacy[static_cast<size_t>(i)*4+j]=command[j];raw[static_cast<size_t>(i)*4+j]=command[j]/scale.value[j];}}
    device_ptr<float> d_raw(raw.size()),d_input(input.size()),d_mean(raw.size()),d_grad(gradient.size()),d_std(4),d_action(raw.size()),d_reference(raw.size()),d_logp(n),d_reference_logp(n);
    d_raw.copy_from_host(raw.data(),raw.size());d_input.copy_from_host(input.data(),input.size());d_grad.copy_from_host(gradient.data(),gradient.size());
    const float deviations[4]={-1.5f,-1.5f,-1.5f,-1.5f};d_std.copy_from_host(deviations,4);rl::SwimmerStream stream;
    shelter_motor::policy_means(d_raw.get(),d_input.get(),d_mean.get(),n,scale,stream.value);
    shelter_motor::policy_mean_grad(d_grad.get(),d_input.get(),n,scale,stream.value);
    rl::sample_gaussian(d_mean.get(),d_std.get(),n,4,71,2,d_action.get(),d_logp.get(),stream.value);
    device_ptr<float> d_legacy(legacy.size());d_legacy.copy_from_host(legacy.data(),legacy.size());
    rl::sample_gaussian(d_legacy.get(),d_std.get(),n,4,71,2,d_reference.get(),d_reference_logp.get(),stream.value);
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream.value));
    std::vector<float> means(raw.size()),grad(raw.size()),actions(raw.size()),original(raw.size()),logp(n),original_logp(n);
    d_mean.copy_to_host(means.data(),means.size());d_grad.copy_to_host(grad.data(),grad.size());d_action.copy_to_host(actions.data(),actions.size());
    d_reference.copy_to_host(original.data(),original.size());d_logp.copy_to_host(logp.data(),logp.size());d_reference_logp.copy_to_host(original_logp.data(),original_logp.size());
    float minimum=1,maximum=-1;int unclipped=0;
    for(int i=0;i<n;++i) for(int j=0;j<4;++j) {
        const size_t k=static_cast<size_t>(i)*4+j;const bool cover=i%2!=0;
        CHECK(means[k]==shelter_motor::action_mean(raw[k],cover,j,scale));CHECK(grad[k]==(cover?1.f:scale.value[j]));
        CHECK(means[k]==(cover?legacy[k]/scale.value[j]:legacy[k]));
        if(!cover) {CHECK(actions[k]==original[k]);CHECK(logp[static_cast<size_t>(i)]==original_logp[static_cast<size_t>(i)]);}
        if(cover && j==2) {
            const float applied=deep::clamp(actions[k],-1,1);minimum=std::min(minimum,applied);maximum=std::max(maximum,applied);
            unclipped+=fabsf(actions[k])<1?1:0;CHECK(deep::clamp(original[k],-1,1)==1.f);
            CHECK(fabsf((actions[k]-means[k])-(original[k]-legacy[k]))<1e-5f);
        }
    }
    CHECK(maximum-minimum>.3f);CHECK(unclipped>32);
}
GPU_TEST_CASE("shelter action PPO raw-mean gradient agrees with finite differences") {
    const std::vector<float> raw={.2f,-.1f,.78125f,-.125f};
    const std::vector<float> deviations={-1.4f,-1.6f,-1.2f,-1.8f};
    shelter_motor::ActuatorScale scale;scale.value[0]=2;scale.value[1]=1;scale.value[2]=128;scale.value[3]=4;
    for(bool cover:{false,true}) {
    std::vector<float> input(shelter_motor::kObs,0.f);input[55]=cover?1.f:0.f;
    float mean[4];std::vector<float> action(4);const float error[4]={.1f,.2f,-.18f,.075f};
    for(int j=0;j<4;++j) {mean[j]=shelter_motor::action_mean(raw[static_cast<size_t>(j)],cover,j,scale);action[static_cast<size_t>(j)]=mean[j]+error[j];}
    const float old_logp=rl::gaussian_logp(mean,deviations.data(),action.data(),4);
    device_ptr<float> d_raw(4),d_input(input.size()),d_mean(4),d_std(4),d_value(1),d_action(4),d_old(1),d_adv(1),d_return(1),d_adv_stats(2),d_gradient(4),d_std_gradient(4),d_value_gradient(1),d_rows(rl::kGaussianStats),d_stats(rl::kGaussianStats);
    const float zero=0,one=1,adv_stats[2]={0,1};
    d_raw.copy_from_host(raw.data(),4);d_input.copy_from_host(input.data(),input.size());d_std.copy_from_host(deviations.data(),4);d_value.copy_from_host(&zero,1);
    d_action.copy_from_host(action.data(),4);d_old.copy_from_host(&old_logp,1);d_adv.copy_from_host(&one,1);d_return.copy_from_host(&zero,1);d_adv_stats.copy_from_host(adv_stats,2);
    CUDA_CHECK_THROW(cudaMemset(d_stats.get(),0,d_stats.size_bytes()));rl::SwimmerStream stream;
    shelter_motor::policy_means(d_raw.get(),d_input.get(),d_mean.get(),1,scale,stream.value);
    rl::gaussian_ppo_loss_grad(d_mean.get(),d_std.get(),d_value.get(),1,4,d_action.get(),d_old.get(),d_adv.get(),d_return.get(),d_adv_stats.get(),{.2f,0.f,.5f},d_gradient.get(),d_std_gradient.get(),d_value_gradient.get(),d_rows.get(),d_stats.get(),stream.value);
    shelter_motor::policy_mean_grad(d_gradient.get(),d_input.get(),1,scale,stream.value);CUDA_CHECK_THROW(cudaStreamSynchronize(stream.value));
    float actual[4];d_gradient.copy_to_host(actual,4);
    const auto loss=[&](int changed,double delta) {
        double mu[4],sigma[4],a[4],dm[4],ds[4],dv=0;
        for(int j=0;j<4;++j) {mu[j]=(static_cast<double>(raw[static_cast<size_t>(j)])+(j==changed?delta:0.0))*(cover?1.0:static_cast<double>(scale.value[j]));sigma[j]=deviations[static_cast<size_t>(j)];a[j]=action[static_cast<size_t>(j)];}
        return rl::ppo_gaussian_row(mu,sigma,4,a,static_cast<double>(old_logp),1.0,0.0,0.0,{.2f,0.f,.5f},1.0,dm,ds,&dv).pg_loss;
    };
    for(int j=0;j<4;++j) {const double epsilon=1e-6,finite=(loss(j,epsilon)-loss(j,-epsilon))/(2*epsilon);CHECK(std::fabs(finite-actual[j])<2e-4*(1+std::fabs(finite)));}
    }
}
