#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include "common.cuh"          // RNG / util
#include "device_types.cuh"    // EnvState POD
#include "step_kernel.cuh"     // kernel prototype

constexpr int NUM_ENVS = 8;
constexpr int OBS    = 4;      // dummy obs‑dim

__global__ void init_state(float* obs, uint8_t* done) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < NUM_ENVS) {
        for (int j = 0; j < OBS; ++j) obs[i * OBS + j] = 0.f;
        done[i] = 0;
    }
}

int main() {
    float   *d_obs, *d_next_obs, *d_reward;
    uint8_t *d_done;
    cudaMalloc(&d_obs,       NUM_ENVS * OBS * sizeof(float));
    cudaMalloc(&d_next_obs,  NUM_ENVS * OBS * sizeof(float));
    cudaMalloc(&d_reward,    NUM_ENVS * sizeof(float));
    cudaMalloc(&d_done,      NUM_ENVS * sizeof(uint8_t));

    init_state<<<1, 32>>>(d_obs, d_done);

    // one batch step with noop actions (nullptr for brevity)
    step_kernel<<<1, 32>>>(d_obs,
                           d_next_obs,
                           d_reward,
                           d_done,
                           /*num_envs=*/NUM_ENVS,
                           /*actions=*/nullptr);

    cudaDeviceSynchronize();

    uint8_t h_done[NUM_ENVS];
    cudaMemcpy(h_done, d_done, NUM_ENVS, cudaMemcpyDeviceToHost);

    for (int i = 0; i < NUM_ENVS; ++i) {
        assert(h_done[i] == 0 && "no terminal flags expected after first step");
    }

    puts("PASS: step_kernel functional ✅");
    cudaFree(d_obs); cudaFree(d_next_obs); cudaFree(d_reward); cudaFree(d_done);
    return 0;
}
