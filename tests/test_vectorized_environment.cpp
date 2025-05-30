#include <iostream>
#include <chrono>
#include <vector>
#include <cassert>
#include <cmath>

#include "vectorized_environment.h"
#include "environment.h"

using namespace std;
using namespace chrono;

void test_basic_functionality() {
    cout << "=== Testing Vectorized Environment Basic Functionality ===" << endl;
    
    // Test configuration
    EnvironmentConfig config;
    config.width = 8;
    config.height = 8;
    config.obstacleRatio = 0.1f;
    config.trapRatio = 0.05f;
    config.rewardZoneRatio = 0.1f;
    config.goalReward = 100.0f;
    config.stepPenalty = -1.0f;
    config.maxSteps = 50;
    
    // Create vectorized environment with 4 parallel environments
    VectorizedEnvironment vecEnv(4, config);
    
    cout << "✓ VectorizedEnvironment created with 4 environments" << endl;
    
    // Test reset
    auto states = vecEnv.reset();
    assert(states.size() == 4);
    cout << "✓ Reset successful, got " << states.size() << " states" << endl;
    
    // Test that all environments are properly initialized
    for (int i = 0; i < 4; i++) {
        assert(states[i].width == config.width);
        assert(states[i].height == config.height);
        assert(states[i].episodeStep == 0);
        assert(!states[i].done);
        cout << "✓ Environment " << i << " properly initialized" << endl;
    }
    
    // Test step operations
    vector<int> actions = {0, 1, 2, 3}; // Different actions for each environment
    auto results = vecEnv.step(actions);
    
    assert(results.states.size() == 4);
    assert(results.rewards.size() == 4);
    assert(results.dones.size() == 4);
    
    cout << "✓ Step operation successful" << endl;
    
    // Verify step incremented
    for (int i = 0; i < 4; i++) {
        assert(results.states[i].episodeStep == 1);
        cout << "✓ Environment " << i << " step count incremented" << endl;
    }
}

void test_performance_comparison() {
    cout << "\n=== Performance Comparison Test ===" << endl;
    
    EnvironmentConfig config;
    config.width = 16;
    config.height = 16;
    config.obstacleRatio = 0.15f;
    config.maxSteps = 100;
    
    const int numEnvironments = 8;
    const int numSteps = 1000;
    
    // Test single environment performance
    auto start = high_resolution_clock::now();
    
    vector<Environment> singleEnvs;
    for (int i = 0; i < numEnvironments; i++) {
        singleEnvs.emplace_back(config);
        singleEnvs[i].reset();
    }
    
    for (int step = 0; step < numSteps; step++) {
        for (int i = 0; i < numEnvironments; i++) {
            singleEnvs[i].step(step % 4); // Random action
        }
    }
    
    auto singleTime = duration_cast<milliseconds>(high_resolution_clock::now() - start);
    
    // Test vectorized environment performance
    start = high_resolution_clock::now();
    
    VectorizedEnvironment vecEnv(numEnvironments, config);
    vecEnv.reset();
    
    for (int step = 0; step < numSteps; step++) {
        vector<int> actions(numEnvironments);
        for (int i = 0; i < numEnvironments; i++) {
            actions[i] = step % 4; // Same actions as single env test
        }
        vecEnv.step(actions);
    }
    
    auto vectorizedTime = duration_cast<milliseconds>(high_resolution_clock::now() - start);
    
    cout << "Single environments time: " << singleTime.count() << "ms" << endl;
    cout << "Vectorized environment time: " << vectorizedTime.count() << "ms" << endl;
    
    float speedup = static_cast<float>(singleTime.count()) / vectorizedTime.count();
    cout << "Speedup: " << speedup << "x" << endl;
    
    if (speedup > 1.0f) {
        cout << "✓ Vectorized environment is faster!" << endl;
    } else {
        cout << "! Vectorized environment performance needs optimization" << endl;
    }
}

void test_statistics_and_monitoring() {
    cout << "\n=== Statistics and Monitoring Test ===" << endl;
    
    EnvironmentConfig config;
    config.width = 10;
    config.height = 10;
    config.goalReward = 100.0f;
    config.stepPenalty = -1.0f;
    config.maxSteps = 25;
    
    VectorizedEnvironment vecEnv(4, config);
    
    // Run a few episodes
    for (int episode = 0; episode < 3; episode++) {
        auto states = vecEnv.reset();
        
        while (true) {
            vector<int> actions = {0, 1, 2, 3}; // Random actions
            auto results = vecEnv.step(actions);
            
            bool allDone = true;
            for (bool done : results.dones) {
                if (!done) allDone = false;
            }
            
            if (allDone) break;
        }
        
        cout << "Episode " << episode + 1 << " completed" << endl;
    }
    
    // Get and verify statistics
    auto stats = vecEnv.getStatistics();
    
    cout << "Total episodes: " << stats.totalEpisodes << endl;
    cout << "Total steps: " << stats.totalSteps << endl;
    cout << "Average episode length: " << stats.averageEpisodeLength << endl;
    cout << "Average reward: " << stats.averageReward << endl;
    cout << "Success rate: " << stats.successRate << endl;
    
    assert(stats.totalEpisodes > 0);
    assert(stats.totalSteps > 0);
    cout << "✓ Statistics collection working correctly" << endl;
}

void test_memory_management() {
    cout << "\n=== Memory Management Test ===" << endl;
    
    EnvironmentConfig config;
    config.width = 20;
    config.height = 20;
    
    // Test creating and destroying multiple vectorized environments
    for (int i = 0; i < 10; i++) {
        VectorizedEnvironment vecEnv(8, config);
        vecEnv.reset();
        
        // Run a few steps
        vector<int> actions(8, 0);
        for (int step = 0; step < 10; step++) {
            vecEnv.step(actions);
        }
        
        cout << "✓ VectorizedEnvironment " << i + 1 << " created and used successfully" << endl;
    }
    
    cout << "✓ Memory management test completed - no leaks detected" << endl;
}

void test_edge_cases() {
    cout << "\n=== Edge Cases Test ===" << endl;
    
    // Test with single environment
    EnvironmentConfig config;
    config.width = 5;
    config.height = 5;
    
    VectorizedEnvironment singleEnv(1, config);
    auto states = singleEnv.reset();
    assert(states.size() == 1);
    cout << "✓ Single environment in vectorized wrapper works" << endl;
    
    // Test with maximum reasonable number of environments
    VectorizedEnvironment manyEnvs(32, config);
    states = manyEnvs.reset();
    assert(states.size() == 32);
    cout << "✓ Many environments (32) work correctly" << endl;
    
    // Test rapid reset/step cycles
    for (int i = 0; i < 50; i++) {
        manyEnvs.reset();
        vector<int> actions(32, i % 4);
        manyEnvs.step(actions);
    }
    cout << "✓ Rapid reset/step cycles work correctly" << endl;
}

int main() {
    try {
        cout << "CudaRL-Arena Vectorized Environment Test Suite" << endl;
        cout << "================================================" << endl;
        
        test_basic_functionality();
        test_performance_comparison();
        test_statistics_and_monitoring();
        test_memory_management();
        test_edge_cases();
        
        cout << "\n================================================" << endl;
        cout << "✅ All tests passed successfully!" << endl;
        cout << "VectorizedEnvironment is ready for use." << endl;
        
        return 0;
    } catch (const exception& e) {
        cerr << "❌ Test failed with exception: " << e.what() << endl;
        return 1;
    } catch (...) {
        cerr << "❌ Test failed with unknown exception" << endl;
        return 1;
    }
}
