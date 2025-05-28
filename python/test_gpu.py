#!/usr/bin/env python3
import numpy as np
from numba import cuda


@cuda.jit
def vec_mul2(in_arr, out_arr):
    idx = cuda.grid(1)
    if idx < in_arr.size:
        out_arr[idx] = in_arr[idx] * 2.0

def main():
    # discover devices
    gpus = cuda.gpus
    print(f"Detected {len(gpus)} CUDA GPU(s)")
    if not gpus:
        print("No CUDA devices found!")
        return

    dev = gpus[0]
    print(f"Using device 0: {dev.name}")

    # prepare data
    n = 10_000_000
    h_x = np.arange(n, dtype=np.float32)
    d_x = cuda.to_device(h_x)
    d_y = cuda.device_array(n, dtype=np.float32)

    # launch kernel
    threads_per_block = 256
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
    vec_mul2[blocks_per_grid, threads_per_block](d_x, d_y)

    # copy back and verify
    h_y = d_y.copy_to_host()
    print("first 5 results:", h_y[:5])

if __name__ == "__main__":
    main()  