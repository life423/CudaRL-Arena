// Recover the requested world course from the exact first-rollout inputs sent
// to the GPU actor. Reads recorded bytes only. No world or network is stepped.
#include "core/checkpoint.h"
#include "rl/pond_ppo_task.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace cp = cudarl::checkpoint;
namespace task = cudarl::rl::pond_ppo_task;
namespace p = cudarl::physics;
constexpr int worlds = 512, features = 41;
constexpr double rad_to_deg = 57.29577951308232;

static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
static p::V vec(const float* v) { return {v[0], v[1], v[2]}; }
static double course(p::V v) { return std::atan2(-double(v.z), double(v.x)) * rad_to_deg; }

int main(int argc, char** argv) try {
    require(argc == 2, "Expected the first 512-world, 64-control batch");
    const auto bytes = cp::read(argv[1]);
    cp::Reader r{bytes};
    require(r.value<uint32_t>() == 0x31425450 && r.value<uint32_t>() == 1 &&
            r.value<uint32_t>() == 1 && r.value<int>() == 64, "Wrong batch or update");
    (void)r.value<uint64_t>(); (void)r.value<uint64_t>(); // sampler serial, source identity
    (void)r.array<uint8_t>();
    const auto tasks = r.array<uint8_t>();
    (void)r.array<float>(); (void)r.array<float>(); // weights
    const auto observations = r.array<float>();
    for (int array = 0; array < 8; ++array) (void)r.array<float>();
    (void)r.array<uint64_t>(); (void)r.array<uint64_t>(); (void)r.array<int>();
    const auto stride = r.value<uint32_t>();
    const auto traces = r.array<uint8_t>();
    r.finish();
    require(tasks.size() == worlds*sizeof(task::Task) && observations.size() == 64*worlds*features &&
            stride >= 5*sizeof(float) && traces.size() == size_t(64)*worlds*stride, "Batch layout mismatch");
    int left = 0, right = 0;
    std::array<int,5> pairs{};
    double max_course_error = 0, max_velocity_error = 0, max_horizontal_position_input = 0;
    for (int i = 0; i < worlds; ++i) {
        task::Task initial;
        std::memcpy(&initial, tasks.data()+i*sizeof(initial), sizeof(initial));
        if (initial.kind != task::Turn) continue;
        const double yaw = initial.turn_radians * rad_to_deg;
        if (yaw > 0) ++left; else ++right;
        const int magnitudes[]{5,15,30,60,90};
        for (int k=0; k<5; ++k) if(std::abs(std::abs(yaw)-magnitudes[k])<0.001) ++pairs[k];
        // Control 40 still has the original request. Control 41 has the new one.
        for (int control : {40,41}) {
            const float* o = observations.data()+((control-1)*worlds+i)*features;
            const auto heading=vec(o+6), up=vec(o+9), local_v=vec(o+12);
            const p::V horiz=local_v-up*p::dot(up,local_v);
            float xyz[3];
            // Control k begins at the position/velocity recorded after control k-1.
            std::memcpy(xyz,traces.data()+size_t((control-2)*worlds+i)*stride+2*sizeof(float),sizeof(xyz));
            require(std::hypot(xyz[0],xyz[2])>1e-6 && p::norm(horiz)>1e-6f, "World course undefined");
            const double recovered = course(vec(xyz)) +
                std::atan2(double(p::dot(up,p::cross(horiz,heading))),double(p::dot(horiz,heading)))*rad_to_deg;
            const double expected = course(initial.request.request.heading) + (control==41 ? yaw : 0.0);
            const double error = std::abs(std::remainder(recovered-expected,360.0));
            require(std::isfinite(error), "Nonfinite world-course comparison");
            max_course_error=std::max(max_course_error,error);
            max_velocity_error=std::max(max_velocity_error,double(p::norm(vec(o+3)-heading*0.4f)));
            max_horizontal_position_input=std::max(max_horizontal_position_input,double(p::norm(p::cross(vec(o),up))));
        }
    }
    require(left>0 && right>0 && std::all_of(pairs.begin(),pairs.end(),[](int n){return n>0;}), "Missing directions or magnitudes");
    std::cout<<"{\"recorded_turn_worlds\":"<<left+right<<",\"positive\":"<<left<<",\"negative\":"<<right
             <<",\"max_actor_request_error_deg\":"<<max_course_error
             <<",\"max_scaled_terminal_velocity_error\":"<<max_velocity_error
             <<",\"max_horizontal_position_input\":"<<max_horizontal_position_input
             <<",\"physical_controls\":0,\"optimizer_steps\":0}\n";
    return 0;
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
