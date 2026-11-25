#ifndef CUDARL_HEADING_REQUEST_OBSERVATION_H
#define CUDARL_HEADING_REQUEST_OBSERVATION_H
#include "motor/pond_motor_observation.h"

// Exact schema5 Turn input mapping from pond_ppo_task::observe. This pure
// product adapter owns no request schedule, action choice, physics or reward.
namespace cudarl::heading_request_observation {
namespace obs=pond_motor_observation;
namespace p=physics;
CUDARL_HD inline bool observe(const obs::RequestState& request,const p::BodyState& body,
        const float* held,const float* rate,const float* previous,float* output) {
    if(!obs::observe(request,body,held,rate,previous,output))return false;
    const auto current=obs::motion(body);
    const p::V vertical_error{0,(request.start_com_model.y-current.com_model.y)*p::scale,0};
    const auto body_error=body.orientation.inverse(vertical_error)/obs::path_scale_m;
    if(!p::finite_vector(body_error))return false;
    obs::vector_features(output,0,body_error);
    return true;
}
}
#endif
