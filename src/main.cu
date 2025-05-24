#include <iostream>
#include <chrono>
#include <thread>
#include "environment.h"
#include "environment_bridge.h"
#include "q_learning.cuh"

// Macro for CUDA error checking
#define CHECK_CUDA(call) do { \
    cudaError_t _err = call; \
    if (_err != cudaSuccess) { \
        std::fprintf(stderr, "CUDA Error %d: %s\n", _err, cudaGetErrorString(_err)); \
        throw std::runtime_error("CUDA call failed"); \
    } \
} while(0)

/**
 * Main entry point for the CUDA RL Arena PoC.
 * Demonstrates the environment-agent interaction with CUDA acceleration.
 */
int main() {
    std::cout << "=== CudaRL-Arena: CUDA-accelerated Reinforcement Learning ===" << std::endl;
    
    // Create environment
    std::cout << "Creating environment..." << std::endl;
    Environment env(0);
    
    // Create Q-learning instance
    std::cout << "Creating Q-learning instance..." << std::endl;
    CudaQLearning q_learning(env.getWidth(), env.getHeight(), 4, 0.1f, 0.99f);
    
    // Run a simple demo episode
    std::cout << "\nRunning demo episode..." << std::endl;
    env.reset();
    
    // Track states, actions, rewards for batch update
    std::vector<std::pair<int, int>> states;
    std::vector<int> actions;
    std::vector<float> rewards;
    std::vector<std::pair<int, int>> next_states;
    std::vector<bool> dones;
    
    // Run for a few steps
    for (int step = 0; step < 10; step++) {
        // Get current state
        int x = env.getAgentX();
        int y = env.getAgentY();
        std::cout << "Step " << step << ": Agent at (" << x << "," << y << ")" << std::endl;
        
        // Select action (random for demo)
        int action = rand() % 4;
        std::cout << "  Taking action " << action << std::endl;
        
        // Store current state and action
        states.push_back({x, y});
        actions.push_back(action);
        
        // Take step
        env.step(action);
        
        // Get next state and reward
        int next_x = env.getAgentX();
        int next_y = env.getAgentY();
        float reward = env.getReward();
        bool done = env.isDone();
        
        std::cout << "  New position: (" << next_x << "," << next_y 
                  << "), Reward: " << reward 
                  << ", Done: " << (done ? "true" : "false") << std::endl;
        
        // Store reward, next state, and done flag
        rewards.push_back(reward);
        next_states.push_back({next_x, next_y});
        dones.push_back(done);
        
        // Break if done
        if (done) {
            std::cout << "  Episode complete!" << std::endl;
            break;
        }
        
        // Small delay for demo purposes
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    
    // Perform batch update of Q-values
    std::cout << "\nPerforming batch update of Q-values on GPU..." << std::endl;
    q_learning.update_batch(states, actions, rewards, next_states, dones);
    
    // Show some Q-values
    std::cout << "\nSample Q-values after update:" << std::endl;
    for (int a = 0; a < 4; a++) {
        std::cout << "  Q(0,0," << a << ") = " << q_learning.get_q_value(0, 0, a) << std::endl;
    }
    
    std::cout << "\nDemo complete! For full training, run the Python script:" << std::endl;
    std::cout << "  python src/train.py --episodes 1000" << std::endl;
    
    return 0;
}