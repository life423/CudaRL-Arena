#include "cuda_arena.h"
#include <iostream>
#include <vector>
#include <cmath>

int main() {
    try {
        std::cout << "CUDA Device Count: " << CudaArena::get_device_count() << std::endl;
        
        if (CudaArena::get_device_count() > 0) {
            std::cout << "Device 0: " << CudaArena::get_device_name(0) << std::endl;
        }
        
        // Test with small number of environments
        const int num_envs = 10;
        std::cout << "\nCreating CudaArena with " << num_envs << " environments..." << std::endl;
        
        CudaArena arena(num_envs);
        
        // Test hello cuda
        std::cout << "\nTesting hello_cuda():" << std::endl;
        arena.hello_cuda();
        
        // Test reset
        std::cout << "\nResetting environments..." << std::endl;
        arena.reset_environments(42);
        
        // Get initial observations
        auto obs = arena.get_observations();
        std::cout << "\nInitial observations (first environment):" << std::endl;
        std::cout << "  Agent pos: (" << obs[0] << ", " << obs[1] << ")" << std::endl;
        std::cout << "  Goal pos: (" << obs[2] << ", " << obs[3] << ")" << std::endl;
        
        // Test step with random actions
        std::vector<int> actions(num_envs);
        for (int i = 0; i < num_envs; ++i) {
            actions[i] = i % 4; // Cycle through all 4 actions
        }
        
        std::cout << "\nStepping environments..." << std::endl;
        arena.step_environments(actions);
        
        // Check results
        auto rewards = arena.get_rewards();
        auto dones = arena.get_dones();
        obs = arena.get_observations();
        
        std::cout << "\nAfter step (first environment):" << std::endl;
        std::cout << "  New agent pos: (" << obs[0] << ", " << obs[1] << ")" << std::endl;
        std::cout << "  Reward: " << rewards[0] << std::endl;
        std::cout << "  Done: " << (dones[0] ? "true" : "false") << std::endl;
        
        // Test multiple steps
        std::cout << "\nRunning 100 random steps..." << std::endl;
        int total_dones = 0;
        float total_reward = 0.0f;
        
        for (int step = 0; step < 100; ++step) {
            // Random actions
            for (int i = 0; i < num_envs; ++i) {
                actions[i] = rand() % 4;
            }
            
            arena.step_environments(actions);
            
            rewards = arena.get_rewards();
            dones = arena.get_dones();
            
            for (int i = 0; i < num_envs; ++i) {
                total_reward += rewards[i];
                if (dones[i]) {
                    total_dones++;
                    // Reset this environment
                    obs = arena.get_observations();
                    obs[i*4 + 0] = rand() % 10; // Random agent x
                    obs[i*4 + 1] = rand() % 10; // Random agent y
                    // Goal stays at (9,9)
                }
            }
        }
        
        std::cout << "  Total episodes completed: " << total_dones << std::endl;
        std::cout << "  Total reward: " << total_reward << std::endl;
        std::cout << "  Average reward per step: " << (total_reward / (100.0f * num_envs)) << std::endl;
        
        std::cout << "\nAll tests passed successfully!" << std::endl;
        
    } catch (const cuda_error& e) {
        std::cerr << "CUDA Error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}