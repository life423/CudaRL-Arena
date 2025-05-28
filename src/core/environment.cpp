#include "environment.h"
#include "../gpu/kernels.cuh"
#include <stdexcept>
#include <algorithm>

namespace cudarl {

// Forward declarations for CUDA kernel calls
extern "C" {
    void launch_reset_environment_kernel(EnvironmentState* d_state, float* d_grid, int width, int height);
    void launch_step_environment_kernel(EnvironmentState* d_state, float* d_grid, int action);
}

Environment::Environment(int id, int width, int height)
    : m_envId(id)
    , m_deviceState(1)
    , m_deviceGrid(width * height)
{
    // Initialize host state
    m_state.width = width;
    m_state.height = height;
    m_state.grid = nullptr; // Will be managed by device memory
    m_state.agent_x = 0;
    m_state.agent_y = 0;
    m_state.reward = 0.0f;
    m_state.done = false;
    
    // Initialize grid data
    initializeGrid();
    
    // Sync initial state to device
    syncToDevice();
}

Environment::~Environment() {
    // RAII cleanup handled by CudaMemory destructors
}

Environment::Environment(Environment&& other) noexcept
    : m_envId(other.m_envId)
    , m_state(other.m_state)
    , m_deviceState(std::move(other.m_deviceState))
    , m_deviceGrid(std::move(other.m_deviceGrid))
{
    // Reset moved-from object
    other.m_state = {};
    other.m_envId = -1;
}

Environment& Environment::operator=(Environment&& other) noexcept {
    if (this != &other) {
        m_envId = other.m_envId;
        m_state = other.m_state;
        m_deviceState = std::move(other.m_deviceState);
        m_deviceGrid = std::move(other.m_deviceGrid);
        
        // Reset moved-from object
        other.m_state = {};
        other.m_envId = -1;
    }
    return *this;
}

void Environment::reset() {
    // Reset state
    m_state.agent_x = 0;
    m_state.agent_y = 0;
    m_state.reward = 0.0f;
    m_state.done = false;
    
    // Initialize grid
    initializeGrid();
    
    // Sync to device and launch reset kernel
    syncToDevice();
    launch_reset_environment_kernel(m_deviceState.get(), m_deviceGrid.get(), 
                                  m_state.width, m_state.height);
    
    // Update host state
    updateHostState();
}

void Environment::step(int action) {
    if (m_state.done) {
        return; // Episode already finished
    }
    
    // Launch step kernel on device
    launch_step_environment_kernel(m_deviceState.get(), m_deviceGrid.get(), action);
    
    // Update host state from device
    updateHostState();
}

float Environment::getCellValue(int x, int y) const {
    if (x < 0 || x >= m_state.width || y < 0 || y >= m_state.height) {
        return 0.0f;
    }
    
    // Get value from device grid
    std::vector<float> hostGrid = m_deviceGrid.copyToHost();
    return hostGrid[y * m_state.width + x];
}

std::vector<float> Environment::getGrid() const {
    return m_deviceGrid.copyToHost();
}

void Environment::initializeGrid() {
    std::vector<float> hostGrid(m_state.width * m_state.height, 0.0f);
    
    // Set goal at top-right corner
    hostGrid[(m_state.height - 1) * m_state.width + (m_state.width - 1)] = 1.0f;
    
    // Set agent position
    hostGrid[m_state.agent_y * m_state.width + m_state.agent_x] = 0.5f;
    
    // Copy to device
    m_deviceGrid.copyFromHost(hostGrid.data());
}

void Environment::updateHostState() {
    // Copy state from device to host
    EnvironmentState deviceState;
    cudaMemcpy(&deviceState, m_deviceState.get(), sizeof(EnvironmentState), cudaMemcpyDeviceToHost);
    
    // Update host state (but keep host grid pointer as nullptr since we use device memory)
    m_state.agent_x = deviceState.agent_x;
    m_state.agent_y = deviceState.agent_y;
    m_state.reward = deviceState.reward;
    m_state.done = deviceState.done;
}

void Environment::syncToDevice() {
    // Copy scalar members of host state to device.
    // The m_state.grid is nullptr and should not be used directly by the device state.
    // Instead, the kernel will receive m_deviceGrid.get() as a separate argument.
    // If the kernel *must* use d_state->grid, then that pointer needs to be set on the device.
    CUDA_CHECK(cudaMemcpy(m_deviceState.get(), &m_state, sizeof(EnvironmentState), cudaMemcpyHostToDevice));

    // If kernels are designed to use d_state->grid, explicitly set it on the device:
    // This ensures the EnvironmentState struct on the device has a valid grid pointer.
    float* device_grid_ptr = m_deviceGrid.get(); // Get the actual device pointer to the grid
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<char*>(m_deviceState.get()) + offsetof(EnvironmentState, grid), // Address of 'grid' member in device state
        &device_grid_ptr,                                                               // Address of the host variable holding the device grid pointer
        sizeof(float*),
        cudaMemcpyHostToDevice));
}

} // namespace cudarl
