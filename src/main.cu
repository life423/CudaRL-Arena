#include <cstdio>
#include <cuda_runtime.h>

__global__ void hello_cuda() {
  printf("🔥 Hello from GPU (thread %d)! 🔥\n", threadIdx.x);
}

int main() {
  // launch 4 threads in one block, just for demo
  hello_cuda<<<1,4>>>();
  // wait for GPU to finish and flush prints
  cudaDeviceSynchronize();

  std::puts("✅ Back on the CPU, PoC complete!");
  return 0;
}
