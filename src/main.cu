#include <cstdio>
#include <cuda_runtime.h>

__global__ void hello_cuda() {
    printf("Hello from GPU (thread %d)! \n", threadIdx.x);
}

int main() {
    // launch 4 threads in one block
    hello_cuda<<<1,4>>>();
    cudaDeviceSynchronize();

    std::puts("Back on the CPU, PoC complete!");

    // you can invoke Python here via the C API, e.g.:
    // Py_Initialize();
    // PyRun_SimpleString("import mvp; mvp.main()");
    // Py_Finalize();

    return 0;
}
