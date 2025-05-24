#include <catch2/catch_test_macros.hpp>
#include "../src/core/environment.h"

using namespace cudarl;

TEST_CASE("Environment initialization", "[environment]") {
    // Create environment with default parameters
    Environment env(0);
    
    SECTION("Default dimensions") {
        REQUIRE(env.getWidth() == 10);
        REQUIRE(env.getHeight() == 10);
    }
    
    SECTION("Agent starts in center") {
        REQUIRE(env.getAgentX() == 5);
        REQUIRE(env.getAgentY() == 5);
    }
    
    SECTION("Initial reward is zero") {
        REQUIRE(env.getReward() == 0.0f);
    }
    
    SECTION("Initial state is not done") {
        REQUIRE(env.isDone() == false);
    }
}

TEST_CASE("Environment reset", "[environment]") {
    Environment env(0);
    
    // Take some steps to move away from initial state
    env.step(1); // right
    env.step(1); // right
    
    // Reset
    env.reset();
    
    SECTION("Agent returns to center") {
        REQUIRE(env.getAgentX() == 5);
        REQUIRE(env.getAgentY() == 5);
    }
    
    SECTION("Reward is reset") {
        REQUIRE(env.getReward() == 0.0f);
    }
    
    SECTION("Done flag is reset") {
        REQUIRE(env.isDone() == false);
    }
}

TEST_CASE("Environment step", "[environment]") {
    Environment env(0);
    
    SECTION("Move right") {
        env.step(1); // right
        REQUIRE(env.getAgentX() == 6);
        REQUIRE(env.getAgentY() == 5);
    }
    
    SECTION("Move down") {
        env.step(2); // down
        REQUIRE(env.getAgentX() == 5);
        REQUIRE(env.getAgentY() == 6);
    }
    
    SECTION("Move left") {
        env.step(3); // left
        REQUIRE(env.getAgentX() == 4);
        REQUIRE(env.getAgentY() == 5);
    }
    
    SECTION("Move up") {
        env.step(0); // up
        REQUIRE(env.getAgentX() == 5);
        REQUIRE(env.getAgentY() == 4);
    }
    
    SECTION("Boundary check") {
        // Create small environment
        Environment small_env(1, 3, 3);
        
        // Try to move out of bounds
        small_env.step(1); // right
        REQUIRE(small_env.getAgentX() == 2);
        
        // Try to move out of bounds again
        small_env.step(1); // right (should be blocked)
        REQUIRE(small_env.getAgentX() == 2); // Position shouldn't change
    }
    
    SECTION("Reward calculation") {
        // Step should give small negative reward
        env.step(1);
        REQUIRE(env.getReward() == Approx(-0.01f));
    }
}

TEST_CASE("Environment goal detection", "[environment]") {
    // Create small environment for easier testing
    Environment env(0, 5, 5);
    
    // Reset to ensure agent is in center
    env.reset();
    
    // Move to goal (top-right corner)
    env.step(1); // right
    env.step(1); // right
    env.step(0); // up
    env.step(0); // up
    
    SECTION("Goal detection") {
        REQUIRE(env.isDone() == true);
    }
    
    SECTION("Goal reward") {
        REQUIRE(env.getReward() == Approx(1.0f));
    }
}

TEST_CASE("Environment grid access", "[environment]") {
    Environment env(0, 5, 5);
    
    SECTION("Grid dimensions") {
        std::vector<float> grid = env.getGrid();
        REQUIRE(grid.size() == 25); // 5x5
    }
    
    SECTION("Cell access") {
        // Goal should be at top-right with value 1.0
        float goal_value = env.getCellValue(4, 0);
        REQUIRE(goal_value == Approx(1.0f));
    }
    
    SECTION("Out of bounds access") {
        float out_of_bounds = env.getCellValue(10, 10);
        REQUIRE(out_of_bounds == 0.0f);
    }
}