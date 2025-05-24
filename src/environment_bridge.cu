#include "environment_bridge.h"
#include <sstream>

EnvironmentBridge::EnvironmentBridge(int width, int height) 
    : width(width), height(height) {
    // Create the underlying environment
    env = std::make_unique<Environment>(0, width, height);
}

std::vector<float> EnvironmentBridge::reset() {
    // Reset the environment
    env->reset();
    
    // Return the current observation
    return getObservation();
}

std::tuple<std::vector<float>, float, bool, std::string> EnvironmentBridge::step(int action) {
    // Take a step in the environment
    env->step(action);
    
    // Get the current observation
    std::vector<float> observation = getObservation();
    
    // Get reward and done state
    float reward = env->getReward();
    bool done = env->isDone();
    
    // Create info string
    std::stringstream info;
    info << "agent_pos: (" << env->getAgentX() << "," << env->getAgentY() << ")";
    
    return std::make_tuple(observation, reward, done, info.str());
}

std::vector<float> EnvironmentBridge::getObservation() const {
    // Get the grid as a flattened vector
    return env->getGrid();
}

int EnvironmentBridge::getWidth() const {
    return width;
}

int EnvironmentBridge::getHeight() const {
    return height;
}

std::pair<int, int> EnvironmentBridge::getAgentPosition() const {
    return std::make_pair(env->getAgentX(), env->getAgentY());
}

void EnvironmentBridge::addObstacle(int x, int y) {
    // In a more complete implementation, this would modify the grid
    // to add an obstacle at the specified position
    // For now, we'll just print a message
    std::cout << "Adding obstacle at (" << x << "," << y << ")" << std::endl;
    
    // This would require extending the Environment class to support obstacles
}

void EnvironmentBridge::setGoal(int x, int y) {
    // In a more complete implementation, this would modify the grid
    // to set the goal at the specified position
    // For now, we'll just print a message
    std::cout << "Setting goal at (" << x << "," << y << ")" << std::endl;
    
    // This would require extending the Environment class to support custom goals
}

void EnvironmentBridge::addTrap(int x, int y, float penalty) {
    // In a more complete implementation, this would modify the grid
    // to add a trap at the specified position
    // For now, we'll just print a message
    std::cout << "Adding trap at (" << x << "," << y << ") with penalty " << penalty << std::endl;
    
    // This would require extending the Environment class to support traps
}