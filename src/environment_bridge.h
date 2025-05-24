#pragma once
#include "environment.h"
#include <vector>
#include <memory>

/**
 * EnvironmentBridge class provides a C++ interface that can be exposed to Python
 * This serves as the connection point between the C++ CUDA environment and Python agents
 */
class EnvironmentBridge {
public:
    /**
     * Constructor
     * @param width Width of the grid environment
     * @param height Height of the grid environment
     */
    EnvironmentBridge(int width = 10, int height = 10);
    
    /**
     * Reset the environment to initial state
     * @return Flattened grid representation as observation
     */
    std::vector<float> reset();
    
    /**
     * Take a step in the environment with the given action
     * @param action Action to take (0=up, 1=right, 2=down, 3=left)
     * @return Tuple of (observation, reward, done, info)
     */
    std::tuple<std::vector<float>, float, bool, std::string> step(int action);
    
    /**
     * Get the current grid state
     * @return Flattened grid representation
     */
    std::vector<float> getObservation() const;
    
    /**
     * Get the width of the environment
     */
    int getWidth() const;
    
    /**
     * Get the height of the environment
     */
    int getHeight() const;
    
    /**
     * Get the agent's current position
     * @return Pair of (x, y) coordinates
     */
    std::pair<int, int> getAgentPosition() const;
    
    /**
     * Add an obstacle at the specified position
     * @param x X coordinate
     * @param y Y coordinate
     */
    void addObstacle(int x, int y);
    
    /**
     * Set the goal position
     * @param x X coordinate
     * @param y Y coordinate
     */
    void setGoal(int x, int y);
    
    /**
     * Add a trap at the specified position
     * @param x X coordinate
     * @param y Y coordinate
     * @param penalty Negative reward for stepping on the trap
     */
    void addTrap(int x, int y, float penalty = -1.0f);
    
private:
    std::unique_ptr<Environment> env;
    int width;
    int height;
};