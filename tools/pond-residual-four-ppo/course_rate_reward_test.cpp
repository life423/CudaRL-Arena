#include "course_rate_reward.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace rp=cudarl::residual_four_ppo;
namespace {
constexpr float pi=3.14159265358979323846f;
int assertions=0;
void check(bool condition,const char* label){
    ++assertions;if(!condition)throw std::runtime_error(label);
}
float quality(float pre_deg,float post_deg,float heading_deg,float speed=.008f){
    const float a=pre_deg*pi/180.f,b=post_deg*pi/180.f,h=heading_deg*pi/180.f;
    return rp::course_rate_quality(speed*std::cos(a),-speed*std::sin(a),
        speed*std::cos(b),-speed*std::sin(b),std::cos(h),-std::sin(h));
}
}
int main(){
    try{
        const float corrective=quality(0,.375f,15);
        const float straight=quality(0,0,15);
        const float wrong=quality(0,-.375f,15);
        check(corrective>.9999f&&corrective>straight&&straight>wrong,"Useful corrective course beats no turn and wrong direction");
        check(std::abs(corrective-quality(0,-.375f,-15))<2e-6f,"Left and right corrections mirror");
        check(std::abs(quality(17,17.25f,27)-quality(-17,-17.25f,-27))<2e-6f,"Mirroring is independent of global heading");
        check(quality(23,23,23)>.9999f,"Settled course rewards zero course rate");
        check(quality(23,23.375f,23)<quality(23,23,23),"Settling discourages continued rotation");
        check(quality(0,.5f,30)>.9999f&&quality(0,.5f,90)>.9999f,"Large errors use the ten-degree-per-second reference cap");
        check(quality(179.9f,-179.85f,-179.6f)>.7f,"Course rate wraps across the angle branch cut");
        check(std::abs(quality(179.9f,-179.85f,-179.6f)-quality(-.1f,.15f,.4f))<1e-4f,"Wrapped and unwrapped small turns agree");
        check(rp::course_rate_quality(0,0,.008f,0,1,0)==.5f,"Zero previous displacement has neutral quality");
        check(rp::course_rate_quality(.008f,0,0,0,1,0)==.5f,"Zero next displacement has neutral quality");
        check(quality(0,.375f,15,.0029f)==.5f,"Subthreshold speed cannot claim a meaningful course rate");
        check(rp::course_rate_quality(.008f,0,.008f,0,0,0)==.5f,"Undefined requested heading has neutral quality");
        for(float invalid:{std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity(),
                -std::numeric_limits<float>::infinity(),std::numeric_limits<float>::max()}){
            check(rp::course_rate_quality(invalid,0,.008f,0,1,0)==.5f,"Invalid or overflowing velocity is neutral");
            check(rp::course_rate_quality(.008f,0,.008f,0,invalid,0)==.5f,"Invalid or overflowing heading is neutral");
        }
        for(int heading=-90;heading<=90;heading+=15)for(int change=-5;change<=5;++change){
            const float q=quality(0,float(change)*.1f,float(heading));
            check(std::isfinite(q)&&q>=.5f&&q<=1.f,"Reward quality remains finite and bounded");
        }
        std::cout<<"{\"passed\":true,\"assertions\":"<<assertions
            <<",\"physical_controls\":0,\"optimizer_steps\":0}\n";return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
