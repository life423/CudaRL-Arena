#include <cuda_runtime.h>
#include "physics/flexible_tail.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iterator>

namespace tail = cudarl::flexible_tail;
namespace physics = cudarl::physics;

namespace {
constexpr int substeps = 64;
constexpr float timestep_s = .0025f;

struct Command {
    float targets[2];
};

struct Fixture {
    physics::BodyState initial;
    Command commands[substeps];
};

struct StepRecord {
    physics::BodyState body;
    tail::StepInfo info;
    tail::Snapshot snapshot;
    bool valid;
};

__host__ __device__ void advance(physics::BodyState& body, const Command& command, StepRecord& row) {
    row.valid = tail::step(body, command.targets, timestep_s, row.info);
    row.body = body;
    row.snapshot = tail::snapshot(body);
}

__global__ void collect_device(const Fixture* fixture, StepRecord* records) {
    auto body = fixture->initial;
    for (int i = 0; i < substeps; ++i) {
        advance(body, fixture->commands[i], records[i]);
        if (!records[i].valid) break;
    }
}

Fixture make_fixture() {
    Fixture fixture{};
    fixture.initial.position = {.2f, -2.f, .3f};
    fixture.initial.orientation = physics::axis_rotation({0, 1, 0}, .16f) *
        physics::axis_rotation({0, 0, 1}, -.03f);
    const float velocity[] = {.008f, -.0001f, -.0013f, .04f, -.11f, .06f, .21f, -.14f, .13f, -.18f};
    const float angle[] = {.07f, -.04f, .16f, -.08f};
    std::copy(std::begin(velocity), std::end(velocity), fixture.initial.velocity);
    std::copy(std::begin(angle), std::end(angle), fixture.initial.angle);
    for (int i = 0; i < substeps; ++i) {
        const double phase = 2. * tail::pi * 2.5 * (i + .5) * timestep_s;
        fixture.commands[i].targets[0] = float(.5 * std::sin(phase) + .1);
        fixture.commands[i].targets[1] = float(.13 * std::sin(phase + .8));
    }
    return fixture;
}

double difference(double a, double b) {
    return std::abs(a - b);
}

double difference(physics::V a, physics::V b) {
    return std::max({difference(a.x, b.x), difference(a.y, b.y), difference(a.z, b.z)});
}

double difference(physics::Q a, physics::Q b) {
    return std::max({difference(a.x, b.x), difference(a.y, b.y), difference(a.z, b.z), difference(a.w, b.w)});
}

struct Errors {
    double position = 0, orientation = 0, joint_angle = 0, linear_velocity = 0, angular_rate = 0;
    double com_position = 0, com_velocity = 0, torque = 0, energy_fraction = 0;

    bool has_difference() const {
        return position > 0 || orientation > 0 || joint_angle > 0 || linear_velocity > 0 ||
            angular_rate > 0 || com_position > 0 || com_velocity > 0 || torque > 0 || energy_fraction > 0;
    }

    bool within_limits() const {
        return position <= 2e-5 && orientation <= 2e-5 && joint_angle <= 2e-5 &&
            angular_rate <= 2e-5 && linear_velocity <= 2e-7 && com_position <= 2e-5 * physics::scale &&
            com_velocity <= 2e-7 && torque / physics::torque <= 2e-5 && energy_fraction <= 1;
    }
};

void compare(const StepRecord& cpu, const StepRecord& gpu, Errors& error) {
    error.position = std::max(error.position, difference(cpu.body.position, gpu.body.position));
    error.orientation = std::max(error.orientation, difference(cpu.body.orientation, gpu.body.orientation));
    error.com_position = std::max(error.com_position, difference(cpu.snapshot.com_m, gpu.snapshot.com_m));
    error.com_velocity = std::max(error.com_velocity,
        difference(cpu.snapshot.com_velocity_m_s, gpu.snapshot.com_velocity_m_s));
    for (int i = 0; i < 4; ++i) {
        error.joint_angle = std::max(error.joint_angle, difference(cpu.body.angle[i], gpu.body.angle[i]));
    }
    for (int i = 0; i < physics::dofs; ++i) {
        auto& maximum = i < 3 ? error.linear_velocity : error.angular_rate;
        maximum = std::max(maximum, difference(cpu.body.velocity[i], gpu.body.velocity[i]));
    }
    for (int i = 0; i < 2; ++i) {
        error.torque = std::max(error.torque, difference(cpu.info.torque_Nm[i], gpu.info.torque_Nm[i]));
    }
    const double cpu_energy[] = {cpu.info.root_work_J, cpu.info.elastic_energy_J, cpu.info.drag_loss_J,
        cpu.info.bend_loss_J, cpu.info.kinetic_energy_J};
    const double gpu_energy[] = {gpu.info.root_work_J, gpu.info.elastic_energy_J, gpu.info.drag_loss_J,
        gpu.info.bend_loss_J, gpu.info.kinetic_energy_J};
    for (int i = 0; i < 5; ++i) {
        const double scale = 1e-12 + 1e-4 * std::max(std::abs(cpu_energy[i]), std::abs(gpu_energy[i]));
        error.energy_fraction = std::max(error.energy_fraction, difference(cpu_energy[i], gpu_energy[i]) / scale);
    }
}
} // namespace

#define CHECK_CUDA(call) do { \
    const cudaError_t code = (call); \
    if (code != cudaSuccess) { \
        std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(code)); \
        return 2; \
    } \
} while (0)

int main() {
    const auto started = std::chrono::steady_clock::now();
    const Fixture fixture = make_fixture();
    cudaDeviceProp device{};
    Fixture* device_fixture = nullptr;
    StepRecord* device_rows = nullptr;
    CHECK_CUDA(cudaGetDeviceProperties(&device, 0));
    CHECK_CUDA(cudaMalloc(&device_fixture, sizeof(fixture)));
    CHECK_CUDA(cudaMalloc(&device_rows, substeps * sizeof(StepRecord)));
    CHECK_CUDA(cudaMemcpy(device_fixture, &fixture, sizeof(fixture), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(device_rows, 0, substeps * sizeof(StepRecord)));

    cudaEvent_t start_event, end_event;
    CHECK_CUDA(cudaEventCreate(&start_event));
    CHECK_CUDA(cudaEventCreate(&end_event));
    CHECK_CUDA(cudaEventRecord(start_event));
    collect_device<<<1, 1>>>(device_fixture, device_rows);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(end_event));

    std::array<StepRecord, substeps> cpu{}, gpu{};
    auto body = fixture.initial;
    int cpu_attempted = 0;
    for (int i = 0; i < substeps; ++i) {
        advance(body, fixture.commands[i], cpu[i]);
        ++cpu_attempted;
        if (!cpu[i].valid) break;
    }
    CHECK_CUDA(cudaMemcpy(gpu.data(), device_rows, sizeof(gpu), cudaMemcpyDeviceToHost));
    float kernel_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&kernel_ms, start_event, end_event));

    int valid_matched = 0;
    int first_difference = -1;
    Errors error;
    for (int i = 0; i < substeps; ++i) {
        if (!cpu[i].valid || !gpu[i].valid) break;
        ++valid_matched;
        compare(cpu[i], gpu[i], error);
        if (first_difference < 0 && error.has_difference()) first_difference = i + 1;
    }
    const auto initial_com = tail::snapshot(fixture.initial).com_m;
    const double displacement_m = physics::norm(tail::snapshot(body).com_m - initial_com);
    const bool passed = cpu_attempted == substeps && valid_matched == substeps &&
        displacement_m > 1e-4 && error.within_limits();

    CHECK_CUDA(cudaEventDestroy(end_event));
    CHECK_CUDA(cudaEventDestroy(start_event));
    CHECK_CUDA(cudaFree(device_fixture));
    CHECK_CUDA(cudaFree(device_rows));
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::printf(
        "{\"pass\":%s,\"device\":\"%s\",\"dt_s\":%.4g,\"CPU_attempted\":%d,"
        "\"valid_matched_substeps\":%d,\"first_comparison_difference_substep\":%d,"
        "\"max_position_model_units\":%.12g,\"max_orientation_component\":%.12g,"
        "\"max_joint_angle_rad\":%.12g,\"max_linear_velocity_m_s\":%.12g,\"max_angular_rate_rad_s\":%.12g,"
        "\"max_COM_position_m\":%.12g,\"max_COM_velocity_m_s\":%.12g,\"max_torque_Nm\":%.12g,"
        "\"max_torque_normalized\":%.12g,\"max_energy_tolerance_fraction\":%.12g,"
        "\"CPU_COM_displacement_m\":%.12g,\"GPU_kernel_ms\":%.6g,\"native_wall_s\":%.6g,\"PPO_updates\":0}\n",
        passed ? "true" : "false", device.name, timestep_s, cpu_attempted, valid_matched, first_difference,
        error.position, error.orientation, error.joint_angle, error.linear_velocity, error.angular_rate,
        error.com_position, error.com_velocity, error.torque, error.torque / physics::torque,
        error.energy_fraction, displacement_m, kernel_ms, elapsed);
    return passed ? 0 : 1;
}
