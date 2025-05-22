# CudaRL-Arena

CudaRL-Arena aims to build a GPU-accelerated reinforcement learning game. The
project combines a high-performance C++/CUDA backend with Python based RL
code so we can prototype algorithms quickly while still taking advantage of
the GPU.

The repository currently contains prebuilt binaries (`vector_add.*`) generated
from an early CUDA experiment. They serve as a proof-of-concept for running
CUDA code but aren't needed for running Python experiments. They can be removed
or replaced once a full game environment is available.

## Setup

1. Install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) that
   matches your GPU drivers.
2. Make sure Python 3 is installed. We recommend using a virtual environment to
   manage packages.
3. Clone this repository:

   ```bash
   git clone <repo-url>
   cd CudaRL-Arena
   ```

## Building CUDA Components

The game engine and any heavy numerical work will be implemented in C++ and
CUDA. To compile a `.cu` file you can use `nvcc`:

```bash
nvcc -o my_program my_program.cu
```

This will produce a binary similar to the included `vector_add.exe`. On Linux
use `-o my_program` (without `.exe`). Link against any additional libraries as
needed.

## Running Python Code

Python scripts can interact with the compiled CUDA/C++ code via bindings or by
calling executables. Install any Python dependencies listed in upcoming
`requirements.txt` files, then run your scripts as usual:

```bash
python train.py
```

## Project Status

Right now the repository only contains the demonstration binaries. Future
commits will bring the full C++/CUDA game engine and the reinforcement learning
logic written in Python.