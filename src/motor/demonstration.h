#ifndef CUDARL_MOTOR_DEMONSTRATION_H
#define CUDARL_MOTOR_DEMONSTRATION_H
#include "motor/straight_task.h"
namespace cudarl::demonstration {
namespace task=straight_motor;namespace p=physics;
constexpr unsigned kVersion=1;
constexpr float kFrequency=1.f,kMinimumAmplitude=.05f,kMaximumAmplitude=.65f;
// rad^2*s/m: an initial command-space estimate from the existing long-stroke
// data, not a calibrated inverse fluid law. The proportional correction also
// acts in amplitude-squared space. Both require the physical calibration gate.
constexpr float kFeedforward=28.f,kFeedback=80.f;
struct Label {task::Action action{};float requested_speed=0,forward_speed=0,amplitude=0;};
// Only the same observation available to the NN is used. No hidden integral,
// filtered measurement, future request, teacher clock or private body state.
inline bool label(const float* observation,Label& output){
    if(!observation)return false;
    for(int i=0;i<task::kObservations;++i)if(!p::finite_number(observation[i]))return false;
    const p::V desired{observation[3]*motor::kVelocityScale,observation[4]*motor::kVelocityScale,observation[5]*motor::kVelocityScale};
    const p::V heading{observation[6],observation[7],observation[8]};
    const p::V velocity{observation[12]*motor::kVelocityScale,observation[13]*motor::kVelocityScale,observation[14]*motor::kVelocityScale};
    const float speed=p::norm(desired);
    if(speed<.004f-1e-7f||speed>.008f+1e-7f||fabsf(p::norm(heading)-1)>1e-4f||p::norm(desired-heading*speed)>1e-6f)return false;
    Label next;next.requested_speed=p::clamp(speed,.004f,.008f);next.forward_speed=p::dot(velocity,heading);
    const float square=kFeedforward*next.requested_speed+kFeedback*(next.requested_speed-next.forward_speed);
    next.amplitude=sqrtf(p::clamp(square,kMinimumAmplitude*kMinimumAmplitude,kMaximumAmplitude*kMaximumAmplitude));
    const float command[2]{2*next.amplitude/.7f-1,(kFrequency-.25f)/.625f-1};
    const float latent[2]{atanhf(command[0]),atanhf(command[1])};
    if(!task::action_from_latent(latent,next.action))return false;output=next;return true;
}
}
#endif
