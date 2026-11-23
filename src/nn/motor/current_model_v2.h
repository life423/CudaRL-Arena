#ifndef CUDARL_CURRENT_MODEL_V2_H
#define CUDARL_CURRENT_MODEL_V2_H
#include "motor/movement_model.h"

// Isolated imposed-flow extension of Model 6. The historical source contract,
// geometry, servo, physical limits and exact zero-current execution stay intact.
namespace cudarl::current_model_v2 {
namespace p = physics;
namespace legacy = movement;
constexpr uint32_t kVersion = 2, kGeometryVersion = legacy::kVersion;
using Reference = legacy::Reference;
struct Telemetry : legacy::Telemetry {
    double body_flow_work = 0, water_supply_work = 0, gravity_work = 0;
};
struct FlowTerms {
    float force[10]{};
    double water_quadratic = 0;
};
struct WaterEnergy {
    double dissipation = 0, supply = 0;
};
CUDARL_HD inline bool moving_water(p::V current) {
    return current.x != 0 || current.y != 0 || current.z != 0;
}
CUDARL_HD inline p::System assemble(const p::BodyState& state, p::Parameters fluid = {},
                                    p::V current = {}, FlowTerms* flow = nullptr) {
    if (flow) *flow = {};
    if (!moving_water(current)) return legacy::assemble(state, fluid, true);
    const auto links = legacy::shape(state);
    p::System out;
    for (int b = 0; b < 3; ++b) {
        const auto& l = links.link[b];
        const auto g = legacy::body_geometry(b);
        const auto bias = p::inertial(l, l.alpha) + p::cross(l.omega, p::inertial(l, l.omega));
        for (int i = 0; i < 10; ++i) {
            out.C[i] += l.mass*p::dot(l.linear[i], l.acceleration) + p::dot(l.angular[i], bias);
            for (int j = 0; j < 10; ++j)
                out.M[i][j] += l.mass*p::dot(l.linear[i], l.linear[j]) +
                              p::dot(l.angular[i], p::inertial(l, l.angular[j]));
        }
        // Version 2 fixes a numerical CUDA FMA contraction: round displaced
        // mass and the represented body mass to float32 BEFORE subtraction.
        // Equal density/mass is exactly neutral; this does not adjust mass,
        // volume, gravity, the drag law or any physical coefficient.
        const volatile float displaced_mass_rounded = fluid.density*l.volume;
        const volatile float body_mass_rounded = l.mass;
        const float buoyancy_force = (displaced_mass_rounded-body_mass_rounded)*fluid.gravity;
        p::load(out, l.linear, {0, buoyancy_force, 0});
        const p::V axis[3] = {l.q.rotate({1,0,0}), l.q.rotate({0,1,0}), l.q.rotate({0,0,1})};
        const float projected[3] = {
            b == 0 ? legacy::pi*g.half_height*g.half_width :
                legacy::pi*g.half_width*g.half_width +
                    (b == 2 ? legacy::Geometry::fin_extension*legacy::Geometry::fin_thickness : 0),
            b == 0 ? legacy::pi*g.length*.5f*g.half_width : l.length*2*g.half_width,
            b == 0 ? legacy::pi*g.length*.5f*g.half_height : l.length*2*g.half_height};
        const float Cd[3] = {b == 0 ? .3f : .15f, b == 0 ? .8f : 1.f, b == 0 ? .8f : 1.f};
        const float radius = cbrtf(3*l.volume/(4*legacy::pi));
        const float node[3] = {-.3872983346207417f, 0, .3872983346207417f};
        const float weight[3] = {5.f/18, 4.f/9, 5.f/18};
        for (int sample = 0; sample < 3; ++sample) {
            const auto r = axis[0]*(node[sample]*l.length);
            p::V J[10];
            for (int i = 0; i < 10; ++i) J[i] = l.linear[i] + p::cross(l.angular[i], r);
            const auto relative = p::multiply(J, state.velocity) - current;
            for (int direction = 0; direction < 3; ++direction) {
                const float coefficient = weight[sample]*(6*legacy::pi*fluid.viscosity*radius +
                    .5f*fluid.density*Cd[direction]*projected[direction]*
                        fabsf(p::dot(relative, axis[direction])));
                const float water_axis = p::dot(current, axis[direction]);
                p::resist(out, J, axis[direction], coefficient);
                p::load(out, J, axis[direction]*(coefficient*water_axis));
                if (flow) {
                    for (int i = 0; i < 10; ++i)
                        flow->force[i] += p::dot(J[i], axis[direction]*(coefficient*water_axis));
                    flow->water_quadratic += static_cast<double>(coefficient)*water_axis*water_axis;
                }
            }
        }
        for (int k = 0; k < 3; ++k)
            p::resist(out, l.angular, axis[k], 8*legacy::pi*fluid.viscosity*radius*radius*radius);
    }
    return out;
}

// Reconstruct the same OLD-state drag coefficients at each quadrature point,
// but evaluate energy at the solved velocity. Taking the relative VECTOR
// before projecting avoids subtracting independently rounded aggregate R/F/S
// energies. Every dissipative term is a coefficient times a square; none is
// clipped. Reservoir work is independently integrated from pointwise forces.
CUDARL_HD inline WaterEnergy water_energy(const p::BodyState& state, p::Parameters fluid,
                                           p::V current, const double* velocity) {
    WaterEnergy out;
    const auto links = legacy::shape(state);
    for (int b = 0; b < 3; ++b) {
        const auto& l = links.link[b];const auto g = legacy::body_geometry(b);
        const p::V axis[3] = {l.q.rotate({1,0,0}),l.q.rotate({0,1,0}),l.q.rotate({0,0,1})};
        const float projected[3] = {
            b == 0 ? legacy::pi*g.half_height*g.half_width :
                legacy::pi*g.half_width*g.half_width +
                    (b == 2 ? legacy::Geometry::fin_extension*legacy::Geometry::fin_thickness : 0),
            b == 0 ? legacy::pi*g.length*.5f*g.half_width : l.length*2*g.half_width,
            b == 0 ? legacy::pi*g.length*.5f*g.half_height : l.length*2*g.half_height};
        const float Cd[3] = {b == 0 ? .3f : .15f,b == 0 ? .8f : 1.f,b == 0 ? .8f : 1.f};
        const float radius = cbrtf(3*l.volume/(4*legacy::pi));
        const float node[3] = {-.3872983346207417f,0,.3872983346207417f};
        const float weight[3] = {5.f/18,4.f/9,5.f/18};
        for (int sample = 0; sample < 3; ++sample) {
            const auto r = axis[0]*(node[sample]*l.length);p::V J[10];
            for (int i = 0; i < 10; ++i) J[i] = l.linear[i]+p::cross(l.angular[i],r);
            const auto old_relative = p::multiply(J,state.velocity)-current;
            double vx = -static_cast<double>(current.x),vy = -static_cast<double>(current.y),vz = -static_cast<double>(current.z);
            for (int i = 0; i < 10; ++i) {vx += J[i].x*velocity[i];vy += J[i].y*velocity[i];vz += J[i].z*velocity[i];}
            for (int direction = 0; direction < 3; ++direction) {
                const float coefficient = weight[sample]*(6*legacy::pi*fluid.viscosity*radius +
                    .5f*fluid.density*Cd[direction]*projected[direction]*fabsf(p::dot(old_relative,axis[direction])));
                const double relative_axis = vx*axis[direction].x+vy*axis[direction].y+vz*axis[direction].z;
                const double water_axis = static_cast<double>(current.x)*axis[direction].x+
                    static_cast<double>(current.y)*axis[direction].y+static_cast<double>(current.z)*axis[direction].z;
                out.dissipation += coefficient*relative_axis*relative_axis;
                out.supply -= coefficient*water_axis*relative_axis;
            }
        }
        double wx = 0,wy = 0,wz = 0;
        for (int i = 0; i < 10; ++i) {wx += l.angular[i].x*velocity[i];wy += l.angular[i].y*velocity[i];wz += l.angular[i].z*velocity[i];}
        const float angular_coefficient = 8*legacy::pi*fluid.viscosity*radius*radius*radius;
        for (int k = 0; k < 3; ++k) {
            const double relative_angular = wx*axis[k].x+wy*axis[k].y+wz*axis[k].z;
            out.dissipation += angular_coefficient*relative_angular*relative_angular;
        }
    }
    return out;
}

CUDARL_HD inline bool substep(p::BodyState& body, const Reference& reference, float h,
                               p::Parameters fluid, Telemetry& telemetry, p::V current = {},
                               contact::Arena arena = {}, p::BodyState* attempted = nullptr) {
    telemetry = {};
    if (!moving_water(current)) {
        legacy::Telemetry original;
        const bool advanced = legacy::substep(body, reference, h, fluid, original, true, arena, attempted);
        static_cast<legacy::Telemetry&>(telemetry) = original;
        telemetry.gravity_work = original.external_work;
        return advanced;
    }
    if (attempted) *attempted = body;
    telemetry.failure_code = 1;
    telemetry.minimum_gap = legacy::clearance(body, true, arena);
    if (!(h > 0) || h > .005001f || telemetry.minimum_gap < .01f || !p::finite_vector(current)) return false;
    telemetry.failure_code = 2;
    for (int j = 0; j < 4; ++j)
        if (!p::finite_number(reference.angle[j]) || fabsf(reference.angle[j]) > p::limit ||
            !p::finite_number(reference.rate[j]) || fabsf(reference.rate[j]) > 12 ||
            !p::finite_number(reference.acceleration[j]) || fabsf(reference.acceleration[j]) > 200) return false;
    FlowTerms flow;
    const auto system = assemble(body, fluid, current, &flow);
    double A[10][10]{}, rhs[10]{}, root_rhs[10]{}, root_v[10]{}, desired_v[4]{};
    const double omega = 2*static_cast<double>(legacy::pi)*3, zeta = .9;
    legacy::discrete_system(system, body, h, A, rhs);
    for (int j = 0; j < 4; ++j) {
        const double acceleration = reference.acceleration[j] + omega*omega*(reference.angle[j]-body.angle[j]) +
                                    2*zeta*omega*(reference.rate[j]-body.velocity[6+j]);
        desired_v[j] = body.velocity[6+j] + h*acceleration;
    }
    for (int i = 0; i < 6; ++i) {
        root_rhs[i] = rhs[i];
        for (int j = 0; j < 4; ++j) root_rhs[i] -= A[i][6+j]*desired_v[j];
    }
    telemetry.failure_code = 3;
    if (!legacy::solve(A, root_rhs, root_v, 6)) return false;
    telemetry.failure_code = 4;
    for (int j = 0; j < 4; ++j) {
        double impulse = -rhs[6+j];
        for (int i = 0; i < 6; ++i) impulse += A[6+j][i]*root_v[i];
        for (int i = 0; i < 4; ++i) impulse += A[6+j][6+i]*desired_v[i];
        telemetry.requested[j] = static_cast<float>(impulse/h);
        telemetry.torque[j] = p::clamp(telemetry.requested[j], -p::torque, p::torque);
        if (!p::finite_number(telemetry.requested[j])) return false;
        rhs[6+j] += h*telemetry.torque[j];
    }
    telemetry.failure_code = 5;
    double next[10]{};
    if (!legacy::solve(A, rhs, next, 10)) return false;
    double flow_power = 0;
    for (int i = 0; i < 10; ++i) {
        double product = 0, scale = fabs(rhs[i]) + 1e-15;
        for (int j = 0; j < 10; ++j) {
            product += A[i][j]*next[j];
            scale += fabs(A[i][j]*next[j]);
            telemetry.numerical_loss += .5*(next[i]-body.velocity[i])*system.M[i][j]*(next[j]-body.velocity[j]);
        }
        telemetry.solve_residual = fmaxf(telemetry.solve_residual, static_cast<float>(fabs(product-rhs[i])/scale));
        flow_power += flow.force[i]*next[i];
        telemetry.gravity_work += h*(system.force[i]-flow.force[i])*next[i];
    }
    const auto energy = water_energy(body,fluid,current,next);
    // Independent pointwise reservoir work and nonnegative relative dissipation
    // audit the original float assembled forces; neither is defined from the
    // energy residual. Uniform flow has no angular/vorticity component.
    telemetry.body_flow_work = h*flow_power;
    telemetry.water_supply_work = h*energy.supply;
    telemetry.fluid_loss = h*energy.dissipation;
    telemetry.external_work = telemetry.gravity_work + telemetry.water_supply_work;
    telemetry.failure_code = 6;
    if (telemetry.solve_residual > 1e-8f || telemetry.fluid_loss < -1e-14 ||
        !::isfinite(telemetry.fluid_loss) || !::isfinite(telemetry.external_work)) return false;
    auto candidate = body;
    candidate.position += p::V{static_cast<float>(next[0]), static_cast<float>(next[1]), static_cast<float>(next[2])}*(h/p::scale);
    const p::V angular{static_cast<float>(next[3]), static_cast<float>(next[4]), static_cast<float>(next[5])};
    const float speed = p::norm(angular);
    if (speed > 1e-10f) candidate.orientation = p::axis_rotation(angular/speed, h*speed)*candidate.orientation;
    candidate.orientation.normalize();
    telemetry.failure_code = 7;
    for (int j = 0; j < 4; ++j) {
        candidate.angle[j] += h*static_cast<float>(next[6+j]);
        telemetry.signed_work += h*telemetry.torque[j]*next[6+j];
        telemetry.positive_work += h*fmax(0., telemetry.torque[j]*next[6+j]);
        if (attempted) *attempted = candidate;
        if (fabsf(candidate.angle[j]) > p::limit) return false;
    }
    candidate.work += static_cast<float>(telemetry.positive_work);
    for (int i = 0; i < 10; ++i) candidate.velocity[i] = static_cast<float>(next[i]);
    candidate.travel += h*p::norm({candidate.velocity[0], candidate.velocity[1], candidate.velocity[2]})/p::scale;
    if (attempted) *attempted = candidate;
    telemetry.failure_code = 8;
    telemetry.minimum_gap = fminf(telemetry.minimum_gap, legacy::clearance(candidate, true, arena));
    if (telemetry.minimum_gap < .01f) return false;
    body = candidate;
    telemetry.failure_code = 0;
    return true;
}
} // namespace cudarl::current_model_v2
#endif
