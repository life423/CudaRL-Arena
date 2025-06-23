#include <iostream>
#include <vector>
#include "core/environment.h"
#include "core/vectorized_environment.h"

int main() {
    std::cout << "=== C++ Environment Smoke Test ===" << std::endl;
    
    // Test 1: Create single environment
    std::cout << "\n1. Creating single environment (10x10)..." << std::endl;
    cudarl::Environment env(0, 10, 10);
    std::cout << "   ✓ Environment created" << std::endl;
    std::cout << "   Size: " << env.getWidth() << "x" << env.getHeight() << std::endl;
    
    // Test 2: Reset environment
    std::cout << "\n2. Testing reset..." << std::endl;
    env.reset();
    std::cout << "   ✓ Reset complete. Agent at (" 
              << env.getAgentX() << ", " << env.getAgentY() << ")" << std::endl;
    
    // Test 3: Step with action
    std::cout << "\n3. Testing step with action..." << std::endl;
    env.step(0); // Move up
    std::cout << "   ✓ Step complete. Reward: " << env.getReward() 
              << ", Done: " << env.isDone() << std::endl;
    std::cout << "   Agent now at (" << env.getAgentX() << ", " 
              << env.getAgentY() << ")" << std::endl;
    
    // Test 4: Vectorized environment
    std::cout << "\n4. Creating vectorized environment (8 envs)..." << std::endl;
    cudarl::VectorizedEnvironment vec_env(8, 10, 10);
    std::cout << "   ✓ Vectorized environment created" << std::endl;
    
    // Test 5: Vectorized reset
    std::cout << "\n5. Testing vectorized reset..." << std::endl;
    auto observations = vec_env.reset();
    std::cout << "   ✓ Reset " << observations.size() << " environments" << std::endl;
    
    std::cout << "\n✅ All C++ API tests PASSED!" << std::endl;
    return 0;
}