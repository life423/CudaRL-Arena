#if defined(CUDARL_WITH_CUDA) && CUDARL_WITH_CUDA
#error "Contact inspection is offline and CPU-only"
#endif
#ifndef CUDARL_WITH_CUDA
#define CUDARL_WITH_CUDA 0
#endif
#include "envs/swimmer/motor_contact_v2.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

// Read saved frames and evaluate geometry/instantaneous algebra only.
// No integration, inference, optimization, new scenario, or replay is performed.
namespace {
using namespace cudarl;namespace mc=motor_contact_v2;namespace fs=std::filesystem;
void need(bool okay,const std::string& message) {if(!okay) throw std::runtime_error(message);}
struct Analysis {
    float gap=1e6f,head_support=0,head_roll=0,head_up=0,kinetic=0,buoyancy_roll_torque=0,buoyancy_potential_release=0;
    float closest_velocity=0,normal_force_algebra=0;
    int link=-1,endpoint=-1;
};
const char* anatomy(int link) {return link==0?"head_ellipsoid":(link==1?"proximal_tail":"distal_tail_fin");}
Analysis analyze(const deep::BodyState& body,const deep::Physics& physical,float initial_up) {
    Analysis out;const auto shape=deep::shape(body);const auto up=shape.link[0].q.rotate({0,1,0});out.head_up=up.y;out.head_roll=atan2f(up.z,up.y)*57.295779513f;
    const auto n=shape.link[0].q.inverse({0,1,0});out.head_support=sqrtf(.01f*.01f*n.x*n.x+.0064f*.0064f*n.y*n.y+.008f*.008f*n.z*n.z);
    const auto& head=shape.link[0];const float displaced=physical.density*head.volume*physical.gravity;
    const auto offset=head.q.rotate({0,.0004f,0}),torque=deep::cross(offset,{0,displaced,0});out.buoyancy_roll_torque=torque.x;
    out.buoyancy_potential_release=displaced*.0004f*(up.y-initial_up);out.kinetic=deep::energy(body);
    for(int link=0;link<3;++link) {
        const auto& l=shape.link[link];const int count=link==0?1:2;const float weight=1.f/count;
        for(int endpoint=0;endpoint<count;++endpoint) {
            const auto r=mc::support_offset(l,link,{0,1,0},endpoint),point=l.center+r;const float gap=point.y+4.f*deep::scale;
            deep::V J[deep::dofs];for(int i=0;i<deep::dofs;++i) J[i]=l.linear[i]+deep::cross(l.angular[i],r);
            const auto velocity=deep::multiply(J,body.velocity);
            if(gap<out.gap) {out.gap=gap;out.link=link;out.endpoint=endpoint;out.closest_velocity=velocity.y;}
            if(gap<0) out.normal_force_algebra+=-mc::kContactStiffness*weight*gap;
        }
    }
    need(fabsf(out.gap-mc::inspect(body).floor_gap)<=1e-7f,"Offline floor anatomy disagrees with recorded geometry");return out;
}
}
int main(int argc,char** argv) {
    try {
        if(argc!=3) {std::cerr<<"Usage: motor_contact_v2_inspect CONTACT-DIRECTORY NEW-OUTPUT\n";return 2;}
        const fs::path input=fs::absolute(argv[1]),output=fs::absolute(argv[2]);need(!fs::exists(output),"Choose a new offline output directory");
        const auto gate_bytes=checkpoint::read((input/"contact-gate.bin").string()),frame_bytes=checkpoint::read((input/"contact-frames.bin").string());
        const auto gate=mc::ContactGate::read(gate_bytes);const auto frames=mc::read_frames(frame_bytes,gate);fs::create_directories(output);
        std::ofstream trace(output/"rolled-support.csv");trace<<std::setprecision(17)<<"backend,world,dt_s,time_s,recorded_step,head_roll_deg,head_up_y,head_vertical_radius_m,head_y_model,head_vy_m_s,omega_x_rad_s,omega_y_rad_s,omega_z_rad_s,joint0_rad,joint1_rad,joint2_rad,joint3_rad,joint_rate0,joint_rate1,joint_rate2,joint_rate3,floor_gap_m,closest_anatomy,closest_endpoint,closest_point_vy_m_s,recorded_prior_substep_spring_force_N,postpose_spring_force_algebra_N,head_buoyancy_roll_torque_Nm,released_righting_potential_J,kinetic_energy_J,excessive,escape,invalid\n";
        std::ofstream points(output/"floor-support-points.csv");points<<std::setprecision(17)<<"backend,world,time_s,anatomy,endpoint,point_x_m,point_y_m,point_z_m,gap_m,point_vy_m_s,postpose_spring_force_algebra_N,active_approach_damping_N_s_m\n";
        std::ostringstream rows;rows<<std::setprecision(17)<<'[';int selected=0;float initial_up[2][3]{};bool initialized[2][3]{};double initial_torque=0,initial_potential=0;
        for(const auto& frame:frames) {
            if(frame.case_id<15||frame.case_id>17) continue;const int variant=frame.case_id-15;
            if(!initialized[frame.backend][variant]) {
                initial_up[frame.backend][variant]=frame.before.orientation.rotate({0,1,0}).y;initialized[frame.backend][variant]=true;
                const auto a=analyze(frame.before,frame.physical,initial_up[frame.backend][variant]);initial_torque=a.buoyancy_roll_torque;
                const auto head=deep::shape(frame.before).link[0];initial_potential=frame.physical.density*head.volume*frame.physical.gravity*.0004*(1-initial_up[frame.backend][variant]);
            }
            const auto a=analyze(frame.after,frame.physical,initial_up[frame.backend][variant]);
            trace<<frame.backend<<','<<frame.case_id<<','<<frame.h<<','<<frame.time_s<<','<<frame.step<<','<<a.head_roll<<','<<a.head_up<<','<<a.head_support<<','<<frame.after.position.y<<','<<frame.after.velocity[1]<<','<<frame.after.velocity[3]<<','<<frame.after.velocity[4]<<','<<frame.after.velocity[5];
            for(float q:frame.after.angle) trace<<','<<q;for(int j=6;j<10;++j) trace<<','<<frame.after.velocity[j];
            const auto status=mc::inspect(frame.after);trace<<','<<a.gap<<','<<anatomy(a.link)<<','<<a.endpoint<<','<<a.closest_velocity<<','<<frame.contact.floor_force<<','<<a.normal_force_algebra<<','<<a.buoyancy_roll_torque<<','<<a.buoyancy_potential_release<<','<<a.kinetic<<','<<status.excessive<<','<<status.escape<<','<<status.invalid<<'\n';
            const auto shape=deep::shape(frame.after);for(int link=0;link<3;++link) {const auto& l=shape.link[link];const int count=link==0?1:2;const float weight=1.f/count;for(int endpoint=0;endpoint<count;++endpoint) {
                const auto r=mc::support_offset(l,link,{0,1,0},endpoint),point=l.center+r;const float gap=point.y+4.f*deep::scale;deep::V J[deep::dofs];for(int i=0;i<deep::dofs;++i) J[i]=l.linear[i]+deep::cross(l.angular[i],r);const auto velocity=deep::multiply(J,frame.after.velocity);
                points<<frame.backend<<','<<frame.case_id<<','<<frame.time_s<<','<<anatomy(link)<<','<<endpoint<<','<<point.x<<','<<point.y<<','<<point.z<<','<<gap<<','<<velocity.y<<','<<(gap<0?-mc::kContactStiffness*weight*gap:0)<<','<<(gap<0&&velocity.y<0?2*sqrtf(mc::kContactStiffness*l.mass)*weight:0)<<'\n';
            }}
            if(selected++) rows<<',';rows<<"{\"backend\":"<<frame.backend<<",\"world\":"<<frame.case_id<<",\"h\":"<<frame.h<<",\"time_s\":"<<frame.time_s<<",\"roll_deg\":"<<a.head_roll<<",\"head_vertical_radius_m\":"<<a.head_support<<",\"head_vy_m_s\":"<<frame.after.velocity[1]<<",\"floor_gap_m\":"<<a.gap<<",\"anatomy\":\""<<anatomy(a.link)<<"\",\"closest_point_vy_m_s\":"<<a.closest_velocity<<",\"buoyancy_roll_torque_Nm\":"<<a.buoyancy_roll_torque<<",\"righting_potential_release_J\":"<<a.buoyancy_potential_release<<",\"kinetic_energy_J\":"<<a.kinetic<<'}';
        }rows<<']';need(selected==24,"Expected four saved observations for each of six rolled cases");
        std::ostringstream report;report<<std::setprecision(17)<<"{\"offline_only\":true,\"additional_physical_steps\":0,\"inference_or_learning_calls\":0,\"physics_version\":2,\"gate_passed\":"<<(gate.passed()?"true":"false")<<",\"recorded_frames\":"<<frames.size()<<",\"selected_rolled_frames\":"<<selected<<",\"initial_passive_roll_torque_Nm\":"<<initial_torque<<",\"initial_righting_potential_to_upright_J\":"<<initial_potential<<",\"normal_spring_energy_at_tolerance_per_link_J\":"<<.5*mc::kContactStiffness*mc::kOverlapTolerance*mc::kOverlapTolerance<<",\"limitation\":\"Algebraic spring forces at recorded postposes are not the preceding microstep's full implicit contact forces. Only aggregate spring force was recorded for that step; actual per-point damping impulses and intermediate trajectories were not saved. No replay performed.\",\"rows\":"<<rows.str()<<"}\n";
        std::ofstream summary(output/"summary.json");summary<<report.str();summary.flush();trace.flush();points.flush();need(bool(summary)&&bool(trace)&&bool(points),"Offline report write failed");
        need(checkpoint::read((input/"contact-gate.bin").string())==gate_bytes&&checkpoint::read((input/"contact-frames.bin").string())==frame_bytes,"Input evidence changed during offline inspection");std::cout<<"CONTACT_V2_OFFLINE_INSPECTED frames="<<selected<<" initial_roll_torque_uNm="<<initial_torque*1e6<<" initial_righting_potential_uJ="<<initial_potential*1e6<<" physics_steps=0\n";return 0;
    } catch(const std::exception& error) {std::cerr<<"CONTACT_V2_INSPECTION_STOPPED "<<error.what()<<'\n';return 1;}
}
