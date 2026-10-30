#include "world.h"
#include "core/checkpoint.h"
#include <cstring>
#include <stdexcept>

namespace cudarl::simulation {
namespace {
namespace cp=checkpoint;
constexpr uint32_t magic=0x46574C44u,version=1;
void vector(cp::Writer& out,physics::V v) {out.value(v.x);out.value(v.y);out.value(v.z);}
physics::V vector(cp::Reader& in) {return {in.value<float>(),in.value<float>(),in.value<float>()};}
void body(cp::Writer& out,const physics::BodyState& b) {
    vector(out,b.position);out.value(b.orientation.x);out.value(b.orientation.y);out.value(b.orientation.z);out.value(b.orientation.w);
    for(float v:b.angle) out.value(v);for(float v:b.velocity) out.value(v);out.value(b.work);out.value(b.travel);
}
physics::BodyState body(cp::Reader& in) {
    physics::BodyState b;b.position=vector(in);
    b.orientation={in.value<float>(),in.value<float>(),in.value<float>(),in.value<float>()};
    for(float& v:b.angle) v=in.value<float>();for(float& v:b.velocity) v=in.value<float>();
    b.work=in.value<float>();b.travel=in.value<float>();return b;
}
void boolean(cp::Writer& out,bool b) {out.value(uint8_t(b));}
bool boolean(cp::Reader& in) {const auto b=in.value<uint8_t>();if(b>1) throw std::runtime_error("Invalid diagnostic boolean");return b!=0;}
void inspect(cp::Writer& out,const contact::Inspection& p) {
    out.value(p.floor_gap);out.value(p.min_other_gap);out.value(p.min_gap);out.value(p.max_floor_gap);
    for(bool b:{p.contact,p.floor_contact,p.excessive,p.escape,p.invalid}) boolean(out,b);
}
contact::Inspection inspect(cp::Reader& in) {
    contact::Inspection p;p.floor_gap=in.value<float>();p.min_other_gap=in.value<float>();p.min_gap=in.value<float>();p.max_floor_gap=in.value<float>();
    p.contact=boolean(in);p.floor_contact=boolean(in);p.excessive=boolean(in);p.escape=boolean(in);p.invalid=boolean(in);return p;
}
uint64_t contract() {
    cp::Writer w;w.value(contact::kPhysicsVersion);w.value(contact::kOverlapTolerance);w.value(contact::kContactStiffness);
    w.value(contact::kContactDampingRatio);w.value(contact::kSubstep);
    const auto head=physics::body_geometry(0),tail=physics::body_geometry(1),fin=physics::body_geometry(2);
    const auto lo=contact::bounds_min_model();
    for(float f:{physics::scale,physics::limit,physics::torque,head.length,tail.length,fin.length,
        head.half_height,head.half_width,tail.half_height,fin.half_height,-lo.x,-lo.y,-lo.z}) w.value(f);
    const Gains g;for(float f:g.kp) w.value(f);for(float f:g.kd) w.value(f);return cp::checksum(w.data);
}
void validate(const WorldState& s) {
    if(!finite_state(s))
        throw std::runtime_error("Invalid diagnostic physical state");
    for(int j=0;j<4;++j) {
        for(float v:{s.held_targets[j],s.last_actual_torques[j],s.raw_action[j]})
            if(!physics::finite_number(v)) throw std::runtime_error("Nonfinite diagnostic actuator state");
        if(fabsf(s.held_targets[j])>physics::limit||fabsf(s.last_actual_torques[j])>1)
            throw std::runtime_error("Invalid diagnostic actuator state");
    }
    for(float v:s.actual_torques) if(!physics::finite_number(v)||fabsf(v)>1) throw std::runtime_error("Invalid actual torque");
    for(float v:s.requested_torques) if(!physics::finite_number(v)) throw std::runtime_error("Invalid requested torque");
    const auto& t=s.telemetry;
    for(float v:{s.mean_squared_torque,t.minimum_floor_gap,t.minimum_other_gap,t.floor_spring_force_max,t.total_spring_force_max,
        t.inspection.floor_gap,t.inspection.min_other_gap,t.inspection.min_gap,t.inspection.max_floor_gap})
        if(!physics::finite_number(v)) throw std::runtime_error("Invalid diagnostic telemetry");
    // Only a stopped unsafe/invalid world can contain a partial final control.
    // Divide before rounding up, avoiding both 10*controls and substeps+9 overflow.
    const uint64_t remainder=s.substeps%10;
    const bool stopped=s.unsafe||s.invalid;
    const uint64_t expected_controls=s.substeps/10+uint64_t(stopped&&remainder!=0);
    if(s.mean_squared_torque<0||s.mean_squared_torque>1.000001f||t.executed_substeps<0||t.executed_substeps>10||
       t.first_unsafe_substep< -1||t.first_unsafe_substep>10||t.contact_samples<0||
       (!stopped&&remainder!=0)||s.control_steps!=expected_controls||
       s.time_seconds!=static_cast<double>(s.substeps)*static_cast<double>(contact::kSubstep))
        throw std::runtime_error("Invalid diagnostic clock or telemetry");
    if(!s.unsafe&&!s.invalid) {const auto p=contact::inspect(s.body);if(p.invalid||p.excessive||p.escape) throw std::runtime_error("Unmarked unsafe diagnostic state");}
}
}
PhysicalWorld::PhysicalWorld() {reset();}
void PhysicalWorld::reset(const physics::BodyState& initial) {
    const auto proof=contact::inspect(initial);
    if(proof.invalid||proof.excessive||proof.escape) throw std::runtime_error("Diagnostic reset pose is not physically valid");
    WorldState next;next.body=initial;next.current=state_.current;next.actuation=state_.actuation;next.fluid=state_.fluid;
    next.telemetry.inspection=proof;next.telemetry.minimum_floor_gap=proof.floor_gap;next.telemetry.minimum_other_gap=proof.min_other_gap;
    state_=next;
}
void PhysicalWorld::set_current(physics::V current) {
    auto settings=state_;settings.current=current;
    if(!valid_settings(settings))
        throw std::runtime_error("Current must be horizontal and at most 0.75 cm/s");
    state_.current=current;
}
void PhysicalWorld::set_actuation(Actuation mode) {
    if(mode!=Actuation::Torque&&mode!=Actuation::JointTarget) throw std::runtime_error("Unknown diagnostic actuator interface");
    if(state_.actuation!=mode) {
        state_.actuation=mode;
        for(int j=0;j<4;++j) {state_.raw_action[j]=0;state_.held_targets[j]=0;}
    }
}
std::vector<uint8_t> PhysicalWorld::serialize() const {
    validate(state_);cp::Writer out;out.value(magic);out.value(version);out.value(contract());const auto& s=state_;
    body(out,s.body);out.value(s.fluid.density);out.value(s.fluid.viscosity);out.value(s.fluid.gravity);vector(out,s.current);
    out.value(uint32_t(s.actuation));for(float v:s.gains.kp) out.value(v);for(float v:s.gains.kd) out.value(v);
    for(float v:s.held_targets) out.value(v);for(float v:s.last_actual_torques) out.value(v);for(float v:s.raw_action) out.value(v);
    for(float v:s.actual_torques) out.value(v);for(float v:s.requested_torques) out.value(v);out.value(s.mean_squared_torque);
    out.value(s.control_steps);out.value(s.substeps);out.value(s.time_seconds);
    boolean(out,s.paused);boolean(out,s.unsafe);boolean(out,s.invalid);
    const auto& t=s.telemetry;inspect(out,t.inspection);out.value(t.minimum_floor_gap);out.value(t.minimum_other_gap);
    out.value(t.floor_spring_force_max);out.value(t.total_spring_force_max);out.value(t.contact_samples);out.value(t.executed_substeps);out.value(t.first_unsafe_substep);
    out.value(cp::checksum(out.data));return out.data;
}
void PhysicalWorld::restore(const std::vector<uint8_t>& bytes) {
    if(bytes.size()<24) throw std::runtime_error("Not a foundation diagnostic state");
    std::vector<uint8_t> payload(bytes.begin(),bytes.end()-sizeof(uint64_t));
    uint64_t saved_hash=0;std::memcpy(&saved_hash,bytes.data()+payload.size(),sizeof(saved_hash));
    if(cp::checksum(payload)!=saved_hash) throw std::runtime_error("Diagnostic state checksum differs");
    cp::Reader in{payload};
    if(in.value<uint32_t>()!=magic||in.value<uint32_t>()!=version||in.value<uint64_t>()!=contract())
        throw std::runtime_error("Incompatible diagnostic state; learner checkpoints are not accepted");
    WorldState s;s.body=body(in);s.fluid={in.value<float>(),in.value<float>(),in.value<float>()};s.current=vector(in);s.actuation=Actuation(in.value<uint32_t>());
    for(float& v:s.gains.kp) v=in.value<float>();for(float& v:s.gains.kd) v=in.value<float>();
    for(float& v:s.held_targets) v=in.value<float>();for(float& v:s.last_actual_torques) v=in.value<float>();for(float& v:s.raw_action) v=in.value<float>();
    for(float& v:s.actual_torques) v=in.value<float>();for(float& v:s.requested_torques) v=in.value<float>();s.mean_squared_torque=in.value<float>();
    s.control_steps=in.value<uint64_t>();s.substeps=in.value<uint64_t>();s.time_seconds=in.value<double>();
    s.paused=boolean(in);s.unsafe=boolean(in);s.invalid=boolean(in);
    auto& t=s.telemetry;t.inspection=inspect(in);t.minimum_floor_gap=in.value<float>();t.minimum_other_gap=in.value<float>();
    t.floor_spring_force_max=in.value<float>();t.total_spring_force_max=in.value<float>();t.contact_samples=in.value<int>();t.executed_substeps=in.value<int>();t.first_unsafe_substep=in.value<int>();
    in.finish();validate(s);state_=s;
}
void PhysicalWorld::save(const std::string& path) const {checkpoint::write_atomic(path,serialize());}
void PhysicalWorld::load(const std::string& path) {restore(checkpoint::read(path));}
}
