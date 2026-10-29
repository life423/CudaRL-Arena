#include "rl/boulder_source.h"
#include "envs/swimmer/predator.h"
#include "rl/shelter_motor_run.h"
#include "nn/mlp_cpu.h"
#include <cstdio>
#include <limits>

namespace {
using namespace cudarl;
struct Gaps {
    float bed_envelope=1e10f,bed_contact=1e10f,sphere=1e10f,arch=1e10f,surface=1e10f;
    int bed_link=-1;
    float combined() const {return fminf(fminf(bed_envelope,sphere),fminf(arch,surface));}
};
Gaps gaps(const deep::BodyState& body,const deep::Habitat& habitat,deep::Basin basin) {
    Gaps out;const auto shape=deep::shape(body);
    out.sphere=deep::boulder_body_clearance(body,habitat);out.arch=deep::arch_body_clearance(body,habitat);
    for(int b=0;b<3;++b) {
        const auto& link=shape.link[b];const auto axis=link.q.rotate({1,0,0});
        for(int sample=0;sample<deep::arch_axial_samples;++sample) {
            const float f=(sample-4)*.25f;
            const auto point=(link.center+axis*(f*link.length*.5f))/deep::scale;
            const auto gradient=deep::habitat_gradient(point.x,point.z,habitat,basin);
            const float magnitude=sqrtf(1+gradient.x*gradient.x+gradient.z*gradient.z);
            const auto normal=deep::V{-gradient.x,1,-gradient.z}/magnitude;
            const float bed=(point.y-deep::habitat_floor(point.x,point.z,habitat,basin))/magnitude-
                deep::arch_slice_support(link,b,f,normal)/deep::scale;
            if(bed<out.bed_envelope) {out.bed_envelope=bed;out.bed_link=b;}
            out.surface=fminf(out.surface,-point.y-deep::arch_slice_support(link,b,f,{0,1,0})/deep::scale);
        }
        // Reproduce the actual unchanged bed-contact locations/radii, which
        // currently differ from the full distal-fin envelope above.
        for(int sample=0;sample<3;++sample) {
            const auto point=(link.center+axis*((sample-1)*link.length/3.f))/deep::scale;
            const auto gradient=deep::habitat_gradient(point.x,point.z,habitat,basin);
            const float magnitude=sqrtf(1+gradient.x*gradient.x+gradient.z*gradient.z);
            const float radius=(b==0?.0064f:.002f)/deep::scale;
            out.bed_contact=fminf(out.bed_contact,(point.y-deep::habitat_floor(point.x,point.z,habitat,basin))/magnitude-radius);
        }
    }
    return out;
}
void snapshot(int cover,int flow,float raise,int tick,const arch::State& state,deep::V eye) {
    const auto g=gaps(state.pond.body,state.habitat,{2});const auto visible=predator::exposure_detail(state.pond.body,eye,state.habitat,{2});
    const auto forward=state.pond.body.orientation.rotate({1,0,0});
    std::printf("HOLD_ZERO_FRAME cover=%d flow_cm_s=%.2f raise=%.2f t=%.2f head=(%.6f,%.6f,%.6f) yaw=%.5f speed_cm_s=%.6f exposure=%.6f head_visible=%.6f tail_visible=%.6f gap=%.6f bed_envelope=%.6f bed_contact=%.6f bed_link=%d sphere=%.6f arch=%.6f surface=%.6f\n",
        cover,flow*.25f,raise,tick*deep::dt,state.pond.body.position.x,state.pond.body.position.y,state.pond.body.position.z,
        atan2f(-forward.z,forward.x),100.f*deep::norm({state.pond.body.velocity[0],state.pond.body.velocity[1],state.pond.body.velocity[2]}),
        visible.visible,visible.head,visible.distal_tail,g.combined(),g.bed_envelope,g.bed_contact,g.bed_link,g.sphere,g.arch,g.surface);
}
void mean_snapshot(const char* name,int cover,int tick,const shelter_motor::State& world,const float* action) {
    const auto& body=world.state.pond.body;const auto g=gaps(body,world.state.habitat,{2});
    const auto visible=predator::exposure_detail(body,world.request.eye,world.state.habitat,{2});
    const auto forward=body.orientation.rotate({1,0,0});
    deep::V low{1e10f,1e10f,1e10f},high{-1e10f,-1e10f,-1e10f};const auto shape=deep::shape(body);
    for(int b=0;b<3;++b) for(int axial=0;axial<9;++axial) {
        const auto& link=shape.link[b];const float f=(axial-4)*.25f,profile=b==0?sqrtf(fmaxf(0.f,1-f*f)):1.f;
        const float ry=(b==0?.0064f:(b==2?.006f:.0015f))*profile,rz=(b==0?.008f:.0015f)*profile;
        for(int rim=0;rim<8;++rim) {
            const float angle=rim*.7853981633974483f;
            const auto point=(link.center+link.q.rotate({f*link.length*.5f,ry*cosf(angle),rz*sinf(angle)}))/deep::scale;
            low={fminf(low.x,point.x),fminf(low.y,point.y),fminf(low.z,point.z)};
            high={fmaxf(high.x,point.x),fmaxf(high.y,point.y),fmaxf(high.z,point.z)};
        }
    }
    std::printf("HOLD_MEAN_FRAME actor=%s cover=%d t=%.2f head=(%.6f,%.6f,%.6f) yaw=%.6f speed_cm_s=%.6f raw=(%.6f,%.6f,%.6f,%.6f) angles=(%.6f,%.6f,%.6f,%.6f) bounds_low=(%.6f,%.6f,%.6f) bounds_high=(%.6f,%.6f,%.6f) exposure=%.6f head_visible=%.6f tail_visible=%.6f gap=%.6f bed_envelope=%.6f bed_contact=%.6f bed_link=%d sphere=%.6f arch=%.6f surface=%.6f\n",
        name,cover,tick*deep::dt,body.position.x,body.position.y,body.position.z,atan2f(-forward.z,forward.x),
        deep::norm({body.velocity[0],body.velocity[1],body.velocity[2]})*100,
        action[0],action[1],action[2],action[3],body.angle[0],body.angle[1],body.angle[2],body.angle[3],
        low.x,low.y,low.z,high.x,high.y,high.z,visible.visible,visible.head,visible.distal_tail,
        g.combined(),g.bed_envelope,g.bed_contact,g.bed_link,g.sphere,g.arch,g.surface);
}
int mean_diagnostic(const rl::BoulderSource& source,const std::string& path) {
    const auto learned=rl::load_shelter_motor_run(path);
    if(learned.source.fingerprint!=source.fingerprint) throw std::runtime_error("Holding run and diagnostic boulder parent differ");
    const auto inherited=rl::transfer_shelter_motor(source.motor);const deep::V eye{-8,4,-1.25f};
    std::printf("HOLD_MEAN_PROTOCOL seconds=10 current=calm physics=unchanged execution=cpu_float_mlp means=deterministic source55_expanded59=1 source=%llu update=%d worlds=%d trained_during_diagnostic=0\n",
        static_cast<unsigned long long>(source.fingerprint),learned.update,learned.worlds);
    int inherited_safe=0,learned_safe=0;
    for(int actor=0;actor<2;++actor) {
        const char* name=actor==0?"inherited_expanded":"learner";
        nn::CpuMlp<float> model(rl::swimmer_specification(4,shelter_motor::kObs));model.params()=actor==0?inherited:learned.actor;
        for(int cover=0;cover<predator::kCoverCount;++cover) {
            shelter_motor::State world;world.state=source.frame;const auto point=predator::cover(cover,eye,world.state.habitat,{2});
            if(!point.valid) throw std::runtime_error("No legal cover pose for learned-mean diagnostic");
            world.state.pond.body=predator::cover_pose(point);world.state.pond.anchor=world.state.pond.body.position;
            world.state.pond.target=-1;world.state.pond.decision=false;world.state.pond.invalid=false;world.state.contacts=0;
            for(float& previous:world.state.pond.last_action) previous=0;
            world.request.cover=true;world.request.goal.valid=true;world.request.goal.cover=cover;world.request.goal.position=point.position;
            world.request.heading=deep::axis_rotation({0,1,0},point.yaw).rotate({1,0,0});world.request.eye=eye;
            const auto origin=world.state.pond.body.position;
            float minimum_gap=1e10f,minimum_bed=1e10f,minimum_contact=1e10f,minimum_sphere=1e10f;
            float maximum_exposure=0,maximum_raw=0,maximum_speed=0;int saturated=0,streak=0,longest=0;
            for(int tick=0;tick<=200;++tick) {
                float observation[shelter_motor::kObs];shelter_motor::observe(world,observation,{2});model.forward(observation,1);
                float action[4];for(int j=0;j<4;++j) {action[j]=model.output()[j];if(!std::isfinite(action[j])) throw std::runtime_error("Non-finite learned mean action");}
                const auto g=gaps(world.state.pond.body,world.state.habitat,{2});
                world.exposure=predator::exposure(world.state.pond.body,eye,world.state.habitat,{2});
                minimum_gap=fminf(minimum_gap,g.combined());minimum_bed=fminf(minimum_bed,g.bed_envelope);
                minimum_contact=fminf(minimum_contact,g.bed_contact);minimum_sphere=fminf(minimum_sphere,g.sphere);
                maximum_exposure=fmaxf(maximum_exposure,world.exposure);
                maximum_speed=fmaxf(maximum_speed,deep::norm({world.state.pond.body.velocity[0],world.state.pond.body.velocity[1],world.state.pond.body.velocity[2]}));
                if(tick==0) std::printf("HOLD_MEAN_INPUT actor=%s cover=%d goal_local=(%.6f,%.6f,%.6f) relative_velocity=(%.6f,%.6f,%.6f) target_present=%.1f mode=%.1f heading_local=(%.6f,%.6f,%.6f) first_raw=(%.6f,%.6f,%.6f,%.6f)\n",
                    name,cover,observation[0],observation[1],observation[2],observation[33],observation[34],observation[35],observation[36],observation[55],observation[56],observation[57],observation[58],action[0],action[1],action[2],action[3]);
                if(tick>0) {if(world.exposure<=.05f && g.combined()>=.08f) {++streak;longest=streak>longest?streak:longest;} else streak=0;}
                if(tick==0 || tick==1 || tick==2 || tick==5 || tick==10 || tick==20 || tick==40 || tick==100 || tick==200)
                    mean_snapshot(name,cover,tick,world,action);
                if(tick==200) break;
                for(int j=0;j<4;++j) {maximum_raw=fmaxf(maximum_raw,fabsf(action[j]));saturated+=fabsf(action[j])>=1?1:0;}
                (void)arch::advance(world.state,action,{}, {2});arch::replenish(world.state,{2});
                if(!std::isfinite(deep::norm(world.state.pond.body.position))) throw std::runtime_error("Non-finite learned-mean physical state");
            }
            if(longest==200) {if(actor==0) ++inherited_safe;else ++learned_safe;}
            std::printf("HOLD_MEAN_SUMMARY actor=%s cover=%d longest_safe_s=%.2f displacement=%.6f max_exposure=%.6f max_speed_cm_s=%.6f max_abs_raw=%.6f saturated_fraction=%.6f minimum_gap=%.6f bed_envelope=%.6f bed_contact=%.6f sphere=%.6f contacts=%d update=%d\n",
                name,cover,longest*deep::dt,deep::norm(world.state.pond.body.position-origin),maximum_exposure,maximum_speed*100,maximum_raw,
                saturated/800.f,minimum_gap,minimum_bed,minimum_contact,minimum_sphere,world.state.contacts,actor==0?source.motor_update:learned.update);
        }
    }
    std::printf("HOLD_MEAN_COMPLETE cases=8 inherited_full_ten_second_holds=%d learner_full_ten_second_holds=%d trained_during_diagnostic=0 modified_source=0\n",inherited_safe,learned_safe);
    return 0;
}
}
int main(int argc,char** argv) {
    using namespace cudarl;
    try {
        if(argc!=2 && argc!=3) throw std::invalid_argument("Usage: predator_hold_diagnostic <full-boulder-training.bin> [full-shelter-motor-run.bin]");
        const auto source=rl::load_boulder_source(argv[1]);const deep::V eye{-8,4,-1.25f};
        if(argc==3) return mean_diagnostic(source,argv[2]);
        std::printf("HOLD_ZERO_PROTOCOL seconds=10 physics=unchanged torques=zero policy=none reset=diagnostic_copy_only units=model_unit_4cm source=%llu\n",
            static_cast<unsigned long long>(source.fingerprint));
        std::printf("HOLD_ZERO_GEOMETRY straight_length=1.800 head_vertical_radius=.160 distal_fin_radius=.150 bed_tail_contact_radius=.050\n");
        int calm_nominal_safe=0;
        for(int cover=0;cover<predator::kCoverCount;++cover) for(int flow=0;flow<3;++flow) for(int elevation=0;elevation<2;++elevation) {
            auto state=source.frame;const auto candidate=predator::cover(cover,eye,state.habitat,{2});
            if(!candidate.valid) throw std::runtime_error("No legal static cover pose for diagnostic");
            const float raise=elevation*.25f;state.pond.body=predator::cover_pose(candidate);state.pond.body.position.y+=raise;
            state.pond.target=-1;state.pond.invalid=false;state.contacts=0;
            state.pond.anchor=state.pond.body.position;
            for(float& value:state.pond.last_action) value=0;
            const auto original=state.pond.body.position;const float action[4]{};
            const deep::V current{flow==0?0.f:(flow==1?.005f:.0075f),0,0};
            float minimum_gap=1e10f,minimum_bed=1e10f,minimum_contact=1e10f,minimum_sphere=1e10f,minimum_arch=1e10f;
            float maximum_exposure=0,maximum_speed=0;int streak=0,longest=0,negative_bed=0,negative_sphere=0,negative_arch=0;
            snapshot(cover,flow==0?0:(flow==1?2:3),raise,0,state,eye);
            for(int tick=0;tick<=200;++tick) {
                const auto g=gaps(state.pond.body,state.habitat,{2});const float visible=predator::exposure(state.pond.body,eye,state.habitat,{2});
                minimum_gap=fminf(minimum_gap,g.combined());minimum_bed=fminf(minimum_bed,g.bed_envelope);
                minimum_contact=fminf(minimum_contact,g.bed_contact);minimum_sphere=fminf(minimum_sphere,g.sphere);minimum_arch=fminf(minimum_arch,g.arch);
                maximum_exposure=fmaxf(maximum_exposure,visible);
                maximum_speed=fmaxf(maximum_speed,deep::norm({state.pond.body.velocity[0],state.pond.body.velocity[1],state.pond.body.velocity[2]}));
                if(tick>0) {
                    if(visible<=.05f && g.combined()>=.08f) {++streak;longest=streak>longest?streak:longest;} else streak=0;
                    negative_bed+=g.bed_envelope<0?1:0;negative_sphere+=g.sphere<0?1:0;negative_arch+=g.arch<0?1:0;
                }
                if(tick==20 || tick==40 || tick==100 || tick==200) snapshot(cover,flow==0?0:(flow==1?2:3),raise,tick,state,eye);
                if(tick==200) break;
                (void)arch::advance(state,action,current,{2});arch::replenish(state,{2});
                if(!std::isfinite(deep::norm(state.pond.body.position))) throw std::runtime_error("Non-finite zero-torque diagnostic");
            }
            if(flow==0 && elevation==0 && longest==200) ++calm_nominal_safe;
            std::printf("HOLD_ZERO_SUMMARY cover=%d flow_cm_s=%.2f raise=%.2f longest_safe_s=%.2f displacement=%.6f max_exposure=%.6f max_speed_cm_s=%.6f minimum_gap=%.6f bed_envelope=%.6f bed_contact=%.6f sphere=%.6f arch=%.6f negative_bed_ticks=%d negative_sphere_ticks=%d negative_arch_ticks=%d contacts=%d learned=0\n",
                cover,deep::norm(current)*100,raise,longest*deep::dt,deep::norm(state.pond.body.position-original),maximum_exposure,maximum_speed*100,
                minimum_gap,minimum_bed,minimum_contact,minimum_sphere,minimum_arch,negative_bed,negative_sphere,negative_arch,state.contacts);
        }
        std::printf("HOLD_ZERO_COMPLETE cases=24 nominal_calm_full_ten_second_holds=%d trained=0 modified_source=0\n",calm_nominal_safe);
        return 0;
    } catch(const std::exception& error) {std::fprintf(stderr,"HOLD_ZERO_ERROR %s\n",error.what());return 1;}
}
