#include "rl/pond_ppo_task.h"
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace pt=cudarl::rl::pond_ppo_task;
namespace rng=cudarl::rng;
namespace {
uint64_t assertions=0;
void check(bool yes,const char* reason){if(!yes)throw std::runtime_error(reason);++assertions;}
uint32_t bits(float value){uint32_t result=0;std::memcpy(&result,&value,sizeof(result));return result;}
// Literal sampling formulas from committed v2/e013b2a, independent of the new
// sample_request() helper. This is the regression reference for default mode.
pt::RequestSample v2_reference(uint32_t env,uint32_t episode,uint32_t seed){
    pt::RequestSample value;
    const uint32_t mode=rng::bits(seed,rng::hash_u32(env^0x6c9a24d1u),episode)%4u;
    value.kind=mode==0?pt::Hold:(mode==1?pt::Straight:pt::Turn);
    value.start=rng::bits(seed,rng::hash_u32(env^0xa4175b39u),episode)%(value.kind==pt::Hold?6u:4u);
    if(value.kind==pt::Turn){const uint32_t turn=rng::bits(seed,rng::hash_u32(env^0xe31f680bu),episode)%10u;
        const uint32_t magnitude=turn%5u;const float degrees=magnitude==0?5.f:(magnitude==1?15.f:(magnitude==2?30.f:(magnitude==3?60.f:90.f)));
        value.turn_radians=(turn<5?-degrees:degrees)*pt::obs::pi/180.f;}
    return value;
}
bool equal(const pt::RequestSample& a,const pt::RequestSample& b){return a.kind==b.kind&&a.start==b.start&&a.stage==b.stage&&bits(a.turn_radians)==bits(b.turn_radians);}
}
int main(){try{
    uint64_t default_requests=0,paired_requests=0;
    const std::array<uint32_t,3> seeds{0x504f4e44u,7u,4813u};
    for(uint32_t seed:seeds)for(uint32_t episode:{0u,1u,7u}){
        for(uint32_t env=0;env<512;++env){const auto actual=pt::sample_request(env,episode,seed);check(equal(actual,v2_reference(env,episode,seed)),"Default requests differ from frozen v2 formulas");check(equal(actual,pt::sample_request(env,episode,seed,pt::curriculum_off)),"Implicit/explicit off differ");check(equal(actual,pt::sample_request(env,episode,seed,pt::course_direction_v1)),"Reward-only program changed v2 requests");check(equal(actual,pt::sample_request(env,episode,seed,pt::repeated_headings_v1)),"Schedule changed original first requests");++default_requests;}
        uint32_t kinds[3]{};bool magnitudes[3]{};
        const std::array<float,3> allowed=episode==0?std::array<float,3>{7,12,18}:std::array<float,3>{18,24,33};
        for(uint32_t pair=0;pair<256;++pair){const auto left=pt::sample_request(pair*2,episode,seed,pt::paired_gentle_moderate_v1),right=pt::sample_request(pair*2+1,episode,seed,pt::paired_gentle_moderate_v1);
            check(left.kind>=pt::Hold&&left.kind<=pt::Turn,"Invalid task kind");++kinds[left.kind];
            check(left.kind==right.kind&&left.start==right.start,"Pair kind/start mismatch");
            check(equal(left,pt::sample_request(pair*2,episode,seed,pt::paired_gentle_moderate_v1)),"Sampler is not deterministic");
            check(left.start<(left.kind==pt::Hold?6u:4u),"Start outside saved fixtures");
            if(left.kind==pt::Turn){check(left.turn_radians<0&&right.turn_radians>0&&left.turn_radians==-right.turn_radians,"Turn pair is not exact left/right symmetry");check(left.stage==(episode==0?0u:1u)&&left.stage==right.stage,"Episode-birth stage mismatch");
                bool found=false;for(size_t i=0;i<allowed.size();++i)if(bits(right.turn_radians)==bits(allowed[i]*pt::obs::pi/180.f)){found=true;magnitudes[i]=true;}check(found,"Angle outside declared stage");
                for(float held_out:{5.f,15.f,30.f,60.f,90.f})check(right.turn_radians!=held_out*pt::obs::pi/180.f,"Held-out evaluation magnitude sampled for training");
            }else check(left.turn_radians==0&&right.turn_radians==0,"Non-turn request has an angle");
            ++paired_requests;
        }
        check(kinds[pt::Hold]>0&&kinds[pt::Straight]>0&&kinds[pt::Turn]>0,"Hold/Straight/Turn representation missing");for(bool used:magnitudes)check(used,"Declared magnitude never sampled");
    }
    check(pt::sample_request(0,0,0,4).kind<0,"Unknown opt-in program accepted");
    check(pt::direction_factor({0,0,0},{1,0,0})==1.f,"Zero-speed direction factor is not finite/neutral");
    check(pt::direction_factor({1,0,0},{1,0,0})==1.f,"Aligned direction factor changed");
    for(float degrees:{0.f,15.f,30.f,90.f,180.f}){
        const float radians=degrees*pt::obs::pi/180.f;const cudarl::physics::V velocity{cosf(radians),0,sinf(radians)},heading{1,0,0};
        const float progress=cudarl::physics::clamp(velocity.x,0,1),legacy=.35f+.65f*progress;
        for(int kind:{pt::Hold,pt::Straight,pt::Turn})for(bool changed:{false,true})for(uint32_t program:{0u,1u,2u})
            if(program!=pt::course_direction_v1||kind!=pt::Turn||!changed)check(bits(pt::moving_reward_terms(1,progress,velocity,heading,kind,changed,program))==bits(legacy),"Course program changed an excluded reward");
        const float focused=pt::moving_reward_terms(1,progress,velocity,heading,pt::Turn,true,pt::course_direction_v1);
        check(std::isfinite(focused)&&focused>=.35f&&focused<=1.f,"Focused reward is nonfinite/unbounded");
        if(degrees==15)check(focused>.66f&&focused<.67f&&legacy>.97f&&legacy<.99f,"15-degree course reward does not match preregistration");
        if(degrees==90||degrees==180)check(bits(focused)==bits(legacy),"Far-error speed incentive was removed");
    }
    std::cout<<"{\"passed\":true,\"assertions\":"<<assertions<<",\"default_v2_requests_compared\":"<<default_requests<<",\"symmetric_pairs_checked\":"<<paired_requests<<",\"task_version\":"<<pt::version<<",\"physical_controls\":0,\"geometry_queries\":0,\"optimizer_calls\":0,\"CUDA_calls\":0}\n";return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
