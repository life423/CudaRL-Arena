#pragma once
#include <cuda_runtime.h>
#include <vector>

/**
 * Class for GPU-accelerated Q-learning
 */
class CudaQLearning {
public:
    /**
     * Constructor
     * @param width Environment width
     * @param height Environment height
     * @param action_space_size Number of possible actions
     * @param learning_rate Learning rate (alpha)
     * @param discount_factor Discount factor (gamma)
     */
    CudaQLearning(int width, int height, int action_space_size,
                 float learning_rate = 0.1f, float discount_factor = 0.99f);
    
    /**
     * Destructor
     */
    ~CudaQLearning();
    
    /**
     * Update Q-values in batch on the GPU
     * @param states Vector of states (x, y coordinates)
     * @param actions Vector of actions taken
     * @param rewards Vector of rewards received
     * @param next_states Vector of next states
     * @param dones Vector of done flags
     */
    void update_batch(
        const std::vector<std::pair<int, int>>& states,
        const std::vector<int>& actions,
        const std::vector<float>& rewards,
        const std::vector<std::pair<int, int>>& next_states,
        const std::vector<bool>& dones
    );
    
    /**
     * Get the current Q-table
     * @return Flattened Q-table as a vector
     */
    std::vector<float> get_q_table() const;
    
    /**
     * Set the Q-table
     * @param q_table Flattened Q-table as a vector
     */
    void set_q_table(const std::vector<float>& q_table);
    
    /**
     * Get the best action for a state
     * @param x X coordinate
     * @param y Y coordinate
     * @return Best action
     */
    int get_best_action(int x, int y) const;
    
    /**
     * Get the Q-value for a state-action pair
     * @param x X coordinate
     * @param y Y coordinate
     * @param action Action
     * @return Q-value
     */
    float get_q_value(int x, int y, int action) const;
    
private:
    int width;
    int height;
    int action_space_size;
    float learning_rate;
    float discount_factor;
    
    // Host Q-table
    std::vector<float> h_q_table;
    
    // Device Q-table
    float* d_q_table;
    
    // Device memory for batch updates
    int* d_states_x;
    int* d_states_y;
    int* d_actions;
    float* d_rewards;
    int* d_next_states_x;
    int* d_next_states_y;
    bool* d_dones;
    
    // Maximum batch size
    int max_batch_size;
};