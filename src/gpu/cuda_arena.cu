#include "cuda_arena.h"
#include "cuda_memory.h"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <cstdio>
#include <random>

#define CUDA_CHECK(expr) do {                                             \
    cudaError_t _err = (expr);                                            \
    if (_err != cudaSuccess) {                                            \
        fprintf(stderr, "CUDA error %s in %s:%d : %s\n", #expr,           \
                __FILE__, __LINE__, cudaGetErrorString(_err));            \
        std::abort();                                                     \
    }                                                                     \
} while (0)

enum class Action : int { Up = 0, Right = 1, Down = 2, Left = 3 };
constexpr int   kObsLen        = 4;
constexpr float kGridMin       = 0.f;
constexpr float kGridMax       = 9.f;
constexpr float kGoalRadius2   = 1.f;      // squared 1.0
constexpr float kStepPenalty   = -0.1f;
constexpr float kGoalReward    = 10.f;

__device__ __constant__ float kDx[4] = {0.f, 1.f, 0.f,-1.f};
__device__ __constant__ float kDy[4] = {1.f, 0.f,-1.f, 0.f};

__global__ void step_environments_kernel(float* __restrict__ obs,
                                         float* __restrict__ rew,
                                         int*   __restrict__ done,
                                         const int* __restrict__ act,
                                         int n)
{
    const int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= n) return;

    float* o = obs + env * kObsLen;
    float ax = o[0], ay = o[1];
    const float gx = o[2], gy = o[3];

    const int a = act[env];
    ax += kDx[a];
    ay += kDy[a];
    ax = fminf(kGridMax, fmaxf(kGridMin, ax));
    ay = fminf(kGridMax, fmaxf(kGridMin, ay));

    const float dx = ax - gx, dy = ay - gy;
    const bool reached = (dx*dx + dy*dy) < kGoalRadius2;

    rew [env] = reached ? kGoalReward : kStepPenalty;
    done[env] = reached;

    o[0] = ax; o[1] = ay;                // write back
}

__global__ void hello_kernel() {
    printf("Hello from GPU, thread %d\n",
           threadIdx.x + blockIdx.x * blockDim.x);
}

/*-----------------------------------*
 *        CudaArena methods          *
 *-----------------------------------*/
CudaArena::CudaArena(int n) : m_num_envs(n)
{
    allocate_memory();
    reset_environments(1234);
}

CudaArena::~CudaArena()
{
    free_memory();
}

void CudaArena::allocate_memory()
{
    try {
        // Allocate device memory using RAII wrappers
        d_observations.allocate(m_num_envs * kObsLen);
        d_rewards.allocate(m_num_envs);
        d_dones.allocate(m_num_envs);
        d_actions.allocate(m_num_envs);

        // Resize host vectors
        h_observations.resize(m_num_envs * kObsLen);
        h_rewards.resize(m_num_envs);
        h_dones.resize(m_num_envs);
        h_actions.resize(m_num_envs);
    } catch (const cuda_error& e) {
        fprintf(stderr, "Failed to allocate CUDA memory: %s\n", e.what());
        throw;
    }
}

void CudaArena::free_memory()
{
    // RAII handles cleanup automatically when device_ptr objects are destroyed
    // We can explicitly reset them if needed
    d_observations.reset();
    d_rewards.reset();
    d_dones.reset();
    d_actions.reset();
}

void CudaArena::reset_environments(uint32_t seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> pos(kGridMin, kGridMax);

    for (int i = 0; i < m_num_envs; ++i) {
        h_observations[i*kObsLen + 0] = pos(rng);
        h_observations[i*kObsLen + 1] = pos(rng);
        h_observations[i*kObsLen + 2] = kGridMax;
        h_observations[i*kObsLen + 3] = kGridMax;
        h_rewards[i] = 0.f;
        h_dones  [i] = 0;
    }

    try {
        d_observations.copy_from_host(h_observations.data(), m_num_envs * kObsLen);
    } catch (const cuda_error& e) {
        fprintf(stderr, "Failed to copy observations to device: %s\n", e.what());
        throw;
    }
}

void CudaArena::step_environments(const std::vector<int>& actions)
{
    try {
        h_actions = actions;
        d_actions.copy_from_host(h_actions.data(), m_num_envs);

        const int block = 256;
        const int grid  = (m_num_envs + block - 1) / block;

        step_environments_kernel<<<grid, block>>>(d_observations.get(),
            d_rewards.get(), d_dones.get(), d_actions.get(), m_num_envs);
        CUDA_CHECK_THROW(cudaDeviceSynchronize());

        d_observations.copy_to_host(h_observations.data(), m_num_envs * kObsLen);
        d_rewards.copy_to_host(h_rewards.data(), m_num_envs);
        d_dones.copy_to_host(h_dones.data(), m_num_envs);
    } catch (const cuda_error& e) {
        fprintf(stderr, "Failed in step_environments: %s\n", e.what());
        throw;
    }
}

void CudaArena::hello_cuda()
{
    hello_kernel<<<1, 4>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
}

std::vector<float> CudaArena::get_observations() const {
    return h_observations;
}

std::vector<float> CudaArena::get_rewards() const {
    return h_rewards;
}

std::vector<int> CudaArena::get_dones() const {
    return h_dones;
}

int CudaArena::get_device_count() {
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    return device_count;
}

std::string CudaArena::get_device_name(int device_id) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    return std::string(prop.name);
}