#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "../core/vectorized_environment.h"
#include "../core/environment.h"

namespace py = pybind11;

PYBIND11_MODULE(cudarl_vectorized, m) {
    m.doc() = "CudaRL-Arena Vectorized Environment Module";

    // ══════════════════════════════════════════════════════════════════════════════
    // ENUMS AND CONFIGURATION STRUCTURES
    // ══════════════════════════════════════════════════════════════════════════════
    
    py::enum_<cudarl::CellType>(m, "CellType")
        .value("EMPTY", cudarl::CellType::EMPTY)
        .value("AGENT", cudarl::CellType::AGENT)
        .value("GOAL", cudarl::CellType::GOAL)
        .value("OBSTACLE", cudarl::CellType::OBSTACLE)
        .value("TRAP", cudarl::CellType::TRAP)
        .value("REWARD_ZONE", cudarl::CellType::REWARD_ZONE);

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

    py::class_<cudarl::EnvironmentState>(m, "EnvironmentState")
        .def_readonly("width", &cudarl::EnvironmentState::width)
        .def_readonly("height", &cudarl::EnvironmentState::height)
        .def_readonly("agent_x", &cudarl::EnvironmentState::agent_x)
        .def_readonly("agent_y", &cudarl::EnvironmentState::agent_y)
        .def_readonly("reward", &cudarl::EnvironmentState::reward)
        .def_readonly("done", &cudarl::EnvironmentState::done)
        .def_readonly("episode_steps", &cudarl::EnvironmentState::episode_steps)
        .def_readonly("cumulative_reward", &cudarl::EnvironmentState::cumulative_reward)
        .def_readonly("goal_reached", &cudarl::EnvironmentState::goal_reached)
        .def_readonly("obstacles_hit", &cudarl::EnvironmentState::obstacles_hit)
        .def_readonly("traps_triggered", &cudarl::EnvironmentState::traps_triggered)
        .def_readonly("config", &cudarl::EnvironmentState::config);

    // ══════════════════════════════════════════════════════════════════════════════
    // ENHANCED SINGLE ENVIRONMENT
    // ══════════════════════════════════════════════════════════════════════════════

    py::class_<cudarl::Environment>(m, "Environment")
        .def(py::init<int, int, int, const cudarl::EnvironmentConfig&>(),
             py::arg("id") = 0, 
             py::arg("width") = 10, 
             py::arg("height") = 10,
             py::arg("config") = cudarl::EnvironmentConfig{})
        
        // Core methods
        .def("reset", &cudarl::Environment::reset)
        .def("step", &cudarl::Environment::step)
        .def("reset_with_obstacles", &cudarl::Environment::resetWithObstacles,
             py::arg("obstacle_density") = -1.0f)
        .def("is_valid_move", &cudarl::Environment::isValidMove)
        .def("get_cell_type", &cudarl::Environment::getCellType)
        .def("regenerate_environment", &cudarl::Environment::regenerateEnvironment)
        
        // Configuration
        .def("set_environment_config", &cudarl::Environment::setEnvironmentConfig)
        .def("get_environment_config", &cudarl::Environment::getEnvironmentConfig)
        .def("set_obstacle_density", &cudarl::Environment::setObstacleDensity)
        .def("set_reward_parameters", &cudarl::Environment::setRewardParameters)
        .def("enable_advanced_rewards", &cudarl::Environment::enableAdvancedRewards)
        
        // State getters
        .def("get_width", &cudarl::Environment::getWidth)
        .def("get_height", &cudarl::Environment::getHeight)
        .def("get_agent_x", &cudarl::Environment::getAgentX)
        .def("get_agent_y", &cudarl::Environment::getAgentY)
        .def("get_reward", &cudarl::Environment::getReward)
        .def("is_done", &cudarl::Environment::isDone)
        .def("get_episode_steps", &cudarl::Environment::getEpisodeSteps)
        .def("get_cumulative_reward", &cudarl::Environment::getCumulativeReward)
        .def("is_goal_reached", &cudarl::Environment::isGoalReached)
        .def("get_obstacles_hit", &cudarl::Environment::getObstaclesHit)
        .def("get_traps_triggered", &cudarl::Environment::getTrapsTriggered)
        
        // Grid access
        .def("get_cell_value", &cudarl::Environment::getCellValue)
        .def("get_grid", &cudarl::Environment::getGrid)
        .def("get_obstacle_positions", &cudarl::Environment::getObstaclePositions)
        .def("get_trap_positions", &cudarl::Environment::getTrapPositions)
        .def("get_reward_zone_positions", &cudarl::Environment::getRewardZonePositions)
        .def("get_goal_position", &cudarl::Environment::getGoalPosition)
        .def("get_agent_position", &cudarl::Environment::getAgentPosition)
        
        // Enhanced numpy grid access
        .def("get_grid_numpy", [](const cudarl::Environment& env) {
            std::vector<float> grid = env.getGrid();
            return py::array_t<float>(
                {env.getHeight(), env.getWidth()}, // shape
                {sizeof(float) * env.getWidth(), sizeof(float)}, // strides
                grid.data(),
                py::cast(env) // parent to keep data alive
            );
        }, py::return_value_policy::reference_internal)
        
        // Utility methods
        .def("print_grid", &cudarl::Environment::printGrid)
        .def("print_stats", &cudarl::Environment::printStats)
        .def("get_state_string", &cudarl::Environment::getStateString)
        .def("calculate_optimal_path_length", &cudarl::Environment::calculateOptimalPathLength)
        .def("is_reachable", &cudarl::Environment::isReachable)
        .def("get_adjacent_cells", &cudarl::Environment::getAdjacentCells);

    // ══════════════════════════════════════════════════════════════════════════════
    // VECTORIZED ENVIRONMENT
    // ══════════════════════════════════════════════════════════════════════════════

    py::class_<cudarl::VectorizedEnvironment::PerformanceStats>(m, "PerformanceStats")
        .def_readonly("avg_step_time_ms", &cudarl::VectorizedEnvironment::PerformanceStats::avg_step_time_ms)
        .def_readonly("avg_reset_time_ms", &cudarl::VectorizedEnvironment::PerformanceStats::avg_reset_time_ms)
        .def_readonly("total_steps", &cudarl::VectorizedEnvironment::PerformanceStats::total_steps)
        .def_readonly("total_resets", &cudarl::VectorizedEnvironment::PerformanceStats::total_resets)
        .def_readonly("gpu_memory_usage_mb", &cudarl::VectorizedEnvironment::PerformanceStats::gpu_memory_usage_mb);

    py::class_<cudarl::VectorizedEnvironment>(m, "VectorizedEnvironment")
        .def(py::init<int, int, int, float>(),
             py::arg("num_envs") = 8,
             py::arg("width") = 10,
             py::arg("height") = 10,
             py::arg("obstacle_density") = 0.1f)
        
        // Core vectorized operations
        .def("reset", &cudarl::VectorizedEnvironment::reset)
        .def("step", [](cudarl::VectorizedEnvironment& env, const py::array_t<int>& actions) {
            py::buffer_info buf = actions.request();
            if (buf.ndim != 1) {
                throw std::runtime_error("Actions must be 1-dimensional");
            }
            
            std::vector<int> action_vec(static_cast<int*>(buf.ptr), 
                                       static_cast<int*>(buf.ptr) + buf.shape[0]);
            return env.step(action_vec);
        })
        .def("step_single", &cudarl::VectorizedEnvironment::stepSingle)
        .def("reset_single", &cudarl::VectorizedEnvironment::resetSingle)
        
        // Configuration
        .def("regenerate_obstacles", &cudarl::VectorizedEnvironment::regenerateObstacles,
             py::arg("density") = -1.0f)
        .def("set_reward_parameters", &cudarl::VectorizedEnvironment::setRewardParameters,
             py::arg("step_penalty") = -0.01f,
             py::arg("goal_reward") = 1.0f,
             py::arg("obstacle_penalty") = -0.5f,
             py::arg("trap_penalty") = -0.2f)
        .def("set_advanced_rewards", &cudarl::VectorizedEnvironment::setAdvancedRewards)
        
        // Environment info
        .def("get_num_environments", &cudarl::VectorizedEnvironment::getNumEnvironments)
        .def("get_width", &cudarl::VectorizedEnvironment::getWidth)
        .def("get_height", &cudarl::VectorizedEnvironment::getHeight)
        .def("get_obstacle_density", &cudarl::VectorizedEnvironment::getObstacleDensity)
        
        // State access
        .def("get_states", &cudarl::VectorizedEnvironment::getStates)
        .def("get_state", &cudarl::VectorizedEnvironment::getState)
        .def("get_grids", &cudarl::VectorizedEnvironment::getGrids)
        .def("get_grid", &cudarl::VectorizedEnvironment::getGrid)
        .def("get_agent_positions", &cudarl::VectorizedEnvironment::getAgentPositions)
        .def("get_done_flags", &cudarl::VectorizedEnvironment::getDoneFlags)
        .def("get_rewards", &cudarl::VectorizedEnvironment::getRewards)
        
        // Enhanced numpy access
        .def("get_grids_numpy", [](const cudarl::VectorizedEnvironment& env) {
            auto grids = env.getGrids();
            int num_envs = env.getNumEnvironments();
            int height = env.getHeight();
            int width = env.getWidth();
            
            // Create 3D numpy array [num_envs, height, width]
            py::array_t<float> result = py::array_t<float>({num_envs, height, width});
            py::buffer_info buf = result.request();
            float* ptr = static_cast<float*>(buf.ptr);
            
            for (int i = 0; i < num_envs; i++) {
                std::copy(grids[i].begin(), grids[i].end(), ptr + i * height * width);
            }
            
            return result;
        })
        
        .def("get_agent_positions_numpy", [](const cudarl::VectorizedEnvironment& env) {
            auto positions = env.getAgentPositions();
            py::array_t<int> result = py::array_t<int>({(int)positions.size(), 2});
            py::buffer_info buf = result.request();
            int* ptr = static_cast<int*>(buf.ptr);
            
            for (size_t i = 0; i < positions.size(); i++) {
                ptr[i * 2] = positions[i].first;
                ptr[i * 2 + 1] = positions[i].second;
            }
            
            return result;
        })
        
        .def("get_rewards_numpy", [](const cudarl::VectorizedEnvironment& env) {
            auto rewards = env.getRewards();
            return py::array_t<float>(rewards.size(), rewards.data());
        })
        
        .def("get_done_flags_numpy", [](const cudarl::VectorizedEnvironment& env) {
            auto dones = env.getDoneFlags();
            py::array_t<bool> result = py::array_t<bool>(dones.size());
            py::buffer_info buf = result.request();
            bool* ptr = static_cast<bool*>(buf.ptr);
            
            for (size_t i = 0; i < dones.size(); i++) {
                ptr[i] = dones[i];
            }
            
            return result;
        })
        
        // Performance monitoring
        .def("get_performance_stats", &cudarl::VectorizedEnvironment::getPerformanceStats)
        .def("reset_performance_counters", &cudarl::VectorizedEnvironment::resetPerformanceCounters);

    // ══════════════════════════════════════════════════════════════════════════════
    // UTILITY FUNCTIONS
    // ══════════════════════════════════════════════════════════════════════════════
    
    m.def("create_default_config", []() {
        return cudarl::EnvironmentConfig{};
    }, "Create a default environment configuration");
    
    m.def("create_sparse_config", []() {
        cudarl::EnvironmentConfig config;
        config.obstacle_density = 0.05f;
        config.trap_density = 0.02f;
        config.reward_zone_density = 0.01f;
        return config;
    }, "Create a sparse environment configuration with fewer obstacles");
    
    m.def("create_dense_config", []() {
        cudarl::EnvironmentConfig config;
        config.obstacle_density = 0.3f;
        config.trap_density = 0.1f;
        config.reward_zone_density = 0.05f;
        return config;
    }, "Create a dense environment configuration with more obstacles");
    
    m.def("create_challenging_config", []() {
        cudarl::EnvironmentConfig config;
        config.obstacle_density = 0.25f;
        config.trap_density = 0.08f;
        config.reward_zone_density = 0.03f;
        config.step_penalty = -0.02f;
        config.obstacle_penalty = -1.0f;
        config.trap_penalty = -0.5f;
        config.max_episode_steps = 150;
        return config;
    }, "Create a challenging environment configuration");
}
