#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <memory>
#include <stdexcept>
#include "core/environment.h"
#include "core/vectorized_environment.h"

namespace py = pybind11;

// Enhanced PyEnvironment with proper error handling and full feature access
class PyEnvironment {
private:
    std::shared_ptr<cudarl::Environment> env;
    
public:
    PyEnvironment(int width = 10, int height = 10, const cudarl::EnvironmentConfig& config = cudarl::EnvironmentConfig{}) {
        try {
            env = std::make_shared<cudarl::Environment>(0, width, height, config);
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to create environment: " + std::string(e.what()));
        }
    }
    
    py::array_t<float> reset() {
        if (!env) {
            throw std::runtime_error("Environment not initialized");
        }
        
        env->reset();
        return getObservation();
    }
    
    py::tuple step(int action) {
        if (!env) {
            throw std::runtime_error("Environment not initialized");
        }
        
        // Validate action range (0-3 for up, right, down, left)
        if (action < 0 || action > 3) {
            throw std::invalid_argument("Action must be in range [0, 3]");
        }
        
        env->step(action);
        auto obs = getObservation();
        float reward = env->getReward();
        bool done = env->isDone();
        
        // Enhanced info dictionary with full environment state
        py::dict info;
        info["agent_x"] = env->getAgentX();
        info["agent_y"] = env->getAgentY();
        info["episode_steps"] = getEpisodeSteps();
        info["total_rewards_collected"] = getTotalRewardsCollected();
        info["cumulative_reward"] = getCumulativeReward();
        info["goal_reached"] = isGoalReached();
        info["obstacles_hit"] = getObstaclesHit();
        info["traps_triggered"] = getTrapsTriggered();
        info["optimal_path_length"] = env->calculateOptimalPathLength();
        
        return py::make_tuple(obs, reward, done, info);
    }
    
    // Efficient observation using direct CUDA-to-NumPy transfer
    py::array_t<float> getObservation() {
        if (!env) {
            throw std::runtime_error("Environment not initialized");
        }
        
        int height = env->getHeight();
        int width = env->getWidth();
        size_t total_elements = static_cast<size_t>(height) * width;
        
        // Create numpy array with proper shape
        auto result = py::array_t<float>({height, width});
        py::buffer_info buf = result.request();
        float* ptr = static_cast<float*>(buf.ptr);
        
        // Direct CUDA-to-NumPy transfer (most efficient)
        env->copyGridToBuffer(ptr, total_elements);
        
        return result;
    }
    
    // Environment configuration methods
    void setConfig(const cudarl::EnvironmentConfig& config) {
        if (!env) {
            throw std::runtime_error("Environment not initialized");
        }
        env->setEnvironmentConfig(config);
    }
    
    void resetWithObstacles(float obstacle_density = -1.0f) {
        if (!env) {
            throw std::runtime_error("Environment not initialized");
        }
        env->resetWithObstacles(obstacle_density);
    }
    
    void regenerateEnvironment() {
        if (!env) {
            throw std::runtime_error("Environment not initialized");
        }
        env->regenerateEnvironment();
    }
    
    // Enhanced getters with proper error handling
    int getWidth() const { 
        if (!env) throw std::runtime_error("Environment not initialized");
        return env->getWidth(); 
    }
    
    int getHeight() const { 
        if (!env) throw std::runtime_error("Environment not initialized");
        return env->getHeight(); 
    }
    
    int getAgentX() const { 
        if (!env) throw std::runtime_error("Environment not initialized");
        return env->getAgentX(); 
    }
    
    int getAgentY() const { 
        if (!env) throw std::runtime_error("Environment not initialized");
        return env->getAgentY(); 
    }
    
    float getReward() const { 
        if (!env) throw std::runtime_error("Environment not initialized");
        return env->getReward(); 
    }
    
    bool isDone() const { 
        if (!env) throw std::runtime_error("Environment not initialized");
        return env->isDone(); 
    }
    
    py::tuple getAgentPosition() const {
        if (!env) throw std::runtime_error("Environment not initialized");
        return py::make_tuple(env->getAgentX(), env->getAgentY());
    }
    
    // New enhanced state accessors
    int getEpisodeSteps() const {
        if (!env) throw std::runtime_error("Environment not initialized");
        return env->getEpisodeSteps();
    }
    
    int getTotalRewardsCollected() const {
        if (!env) throw std::runtime_error("Environment not initialized");
        return env->getTotalRewardsCollected();
    }
    
    float getCumulativeReward() const {
        if (!env) throw std::runtime_error("Environment not initialized");
        return env->getCumulativeReward();
    }
    
    bool isGoalReached() const {
        if (!env) throw std::runtime_error("Environment not initialized");
        return env->isGoalReached();
    }
    
    int getObstaclesHit() const {
        if (!env) throw std::runtime_error("Environment not initialized");
        return env->getObstaclesHit();
    }
    
    int getTrapsTriggered() const {
        if (!env) throw std::runtime_error("Environment not initialized");
        return env->getTrapsTriggered();
    }
    
    // Environment analysis methods
    std::vector<std::pair<int, int>> getObstaclePositions() const {
        if (!env) throw std::runtime_error("Environment not initialized");
        return env->getObstaclePositions();
    }
    
    std::vector<std::pair<int, int>> getTrapPositions() const {
        if (!env) throw std::runtime_error("Environment not initialized");
        return env->getTrapPositions();
    }
    
    std::vector<std::pair<int, int>> getRewardZonePositions() const {
        if (!env) throw std::runtime_error("Environment not initialized");
        return env->getRewardZonePositions();
    }
    
    std::pair<int, int> getGoalPosition() const {
        if (!env) throw std::runtime_error("Environment not initialized");
        return env->getGoalPosition();
    }
    
    bool isValidMove(int x, int y) const {
        if (!env) throw std::runtime_error("Environment not initialized");
        return env->isValidMove(x, y);
    }
    
    bool isReachable(int target_x, int target_y) const {
        if (!env) throw std::runtime_error("Environment not initialized");
        return env->isReachable(target_x, target_y);
    }
    
    std::string getStateString() const {
        if (!env) throw std::runtime_error("Environment not initialized");
        return env->getStateString();
    }
    
    void printGrid() const {
        if (!env) throw std::runtime_error("Environment not initialized");
        env->printGrid();
    }
    
    void printStats() const {
        if (!env) throw std::runtime_error("Environment not initialized");
        env->printStats();
    }
};

// Enhanced PyVectorizedEnvironment for efficient batch processing
class PyVectorizedEnvironment {
private:
    std::shared_ptr<cudarl::VectorizedEnvironment> vec_env;
    
public:
    PyVectorizedEnvironment(int num_envs, int width = 10, int height = 10, const cudarl::EnvironmentConfig& config = cudarl::EnvironmentConfig{}) {
        try {
            vec_env = std::make_shared<cudarl::VectorizedEnvironment>(num_envs, width, height, config);
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to create vectorized environment: " + std::string(e.what()));
        }
    }
    
    std::vector<py::array_t<float>> reset() {
        if (!vec_env) {
            throw std::runtime_error("Vectorized environment not initialized");
        }
        return vec_env->reset();
    }
    
    py::tuple step(const std::vector<int>& actions) {
        if (!vec_env) {
            throw std::runtime_error("Vectorized environment not initialized");
        }
        return vec_env->step(actions);
    }
    
    int getNumEnvs() const {
        if (!vec_env) throw std::runtime_error("Vectorized environment not initialized");
        return vec_env->getNumEnvs();
    }
    
    int getWidth() const {
        if (!vec_env) throw std::runtime_error("Vectorized environment not initialized");
        return vec_env->getWidth();
    }
    
    int getHeight() const {
        if (!vec_env) throw std::runtime_error("Vectorized environment not initialized");
        return vec_env->getHeight();
    }
};

PYBIND11_MODULE(cudarl_core_python, m) {
    m.doc() = "CudaRL-Arena Python bindings - High-performance CUDA Environment Interface";
    
    // Expose EnvironmentConfig
    py::class_<cudarl::EnvironmentConfig>(m, "EnvironmentConfig")
        .def(py::init<>())
        .def_readwrite("obstacle_density", &cudarl::EnvironmentConfig::obstacle_density)
        .def_readwrite("trap_density", &cudarl::EnvironmentConfig::trap_density)
        .def_readwrite("reward_zone_density", &cudarl::EnvironmentConfig::reward_zone_density)
        .def_readwrite("step_penalty", &cudarl::EnvironmentConfig::step_penalty)
        .def_readwrite("goal_reward", &cudarl::EnvironmentConfig::goal_reward)
        .def_readwrite("obstacle_penalty", &cudarl::EnvironmentConfig::obstacle_penalty)
        .def_readwrite("trap_penalty", &cudarl::EnvironmentConfig::trap_penalty)
        .def_readwrite("reward_zone_bonus", &cudarl::EnvironmentConfig::reward_zone_bonus)
        .def_readwrite("use_advanced_rewards", &cudarl::EnvironmentConfig::use_advanced_rewards)
        .def_readwrite("enable_distance_bonus", &cudarl::EnvironmentConfig::enable_distance_bonus)
        .def_readwrite("max_episode_steps", &cudarl::EnvironmentConfig::max_episode_steps);
    
    // Enhanced Environment class
    py::class_<PyEnvironment>(m, "Environment")
        .def(py::init<int, int, const cudarl::EnvironmentConfig&>(), 
             py::arg("width") = 10, py::arg("height") = 10, py::arg("config") = cudarl::EnvironmentConfig{},
             "Initialize CUDA-accelerated environment with configuration")
        .def("reset", &PyEnvironment::reset,
             "Reset environment and return initial observation")
        .def("step", &PyEnvironment::step, py::arg("action"),
             "Take action and return (observation, reward, done, info)")
        .def("get_observation", &PyEnvironment::getObservation,
             "Get current environment observation (efficient CUDA-to-NumPy)")
        .def("set_config", &PyEnvironment::setConfig, py::arg("config"),
             "Update environment configuration")
        .def("reset_with_obstacles", &PyEnvironment::resetWithObstacles, py::arg("obstacle_density") = -1.0f,
             "Reset with specified obstacle density")
        .def("regenerate_environment", &PyEnvironment::regenerateEnvironment,
             "Regenerate environment with new random layout")
        
        // Basic getters
        .def("get_width", &PyEnvironment::getWidth)
        .def("get_height", &PyEnvironment::getHeight)
        .def("get_agent_x", &PyEnvironment::getAgentX)
        .def("get_agent_y", &PyEnvironment::getAgentY)
        .def("get_reward", &PyEnvironment::getReward)
        .def("is_done", &PyEnvironment::isDone)
        .def("get_agent_position", &PyEnvironment::getAgentPosition)
        
        // Enhanced state getters
        .def("get_episode_steps", &PyEnvironment::getEpisodeSteps)
        .def("get_total_rewards_collected", &PyEnvironment::getTotalRewardsCollected)
        .def("get_cumulative_reward", &PyEnvironment::getCumulativeReward)
        .def("is_goal_reached", &PyEnvironment::isGoalReached)
        .def("get_obstacles_hit", &PyEnvironment::getObstaclesHit)
        .def("get_traps_triggered", &PyEnvironment::getTrapsTriggered)
        
        // Environment analysis
        .def("get_obstacle_positions", &PyEnvironment::getObstaclePositions)
        .def("get_trap_positions", &PyEnvironment::getTrapPositions)
        .def("get_reward_zone_positions", &PyEnvironment::getRewardZonePositions)
        .def("get_goal_position", &PyEnvironment::getGoalPosition)
        .def("is_valid_move", &PyEnvironment::isValidMove, py::arg("x"), py::arg("y"))
        .def("is_reachable", &PyEnvironment::isReachable, py::arg("target_x"), py::arg("target_y"))
        .def("get_state_string", &PyEnvironment::getStateString)
        .def("print_grid", &PyEnvironment::printGrid)
        .def("print_stats", &PyEnvironment::printStats);
    
    // Vectorized Environment for batch processing
    py::class_<PyVectorizedEnvironment>(m, "VectorizedEnvironment")
        .def(py::init<int, int, int, const cudarl::EnvironmentConfig&>(), 
             py::arg("num_envs"), py::arg("width") = 10, py::arg("height") = 10, py::arg("config") = cudarl::EnvironmentConfig{},
             "Initialize batch of CUDA-accelerated environments")
        .def("reset", &PyVectorizedEnvironment::reset,
             "Reset all environments and return list of observations")
        .def("step", &PyVectorizedEnvironment::step, py::arg("actions"),
             "Take actions for all environments and return batch results")
        .def("get_num_envs", &PyVectorizedEnvironment::getNumEnvs)
        .def("get_width", &PyVectorizedEnvironment::getWidth)
        .def("get_height", &PyVectorizedEnvironment::getHeight);
    
    // Constants for action space
    m.attr("ACTION_UP") = 0;
    m.attr("ACTION_RIGHT") = 1;
    m.attr("ACTION_DOWN") = 2;
    m.attr("ACTION_LEFT") = 3;
    
    // Version info
    m.attr("__version__") = "2.0.0";
}
