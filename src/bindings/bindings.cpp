#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Forward declarations for functions we'll implement in other files
// These will be implemented in the AI and game engine components
namespace ai {
    // CUDA-accelerated Q-learning update
    void update_q_table_cuda(float* q_table, int* states, int* actions, float* rewards, int* next_states, 
                            float learning_rate, float discount_factor, int batch_size, int state_space_size, int action_space_size);
    
    // Get best action from Q-table for a given state
    int get_best_action(float* q_table, int state_id, int action_space_size);
    
    // CUDA-accelerated DQN forward pass
    void dqn_forward_cuda(float* states, float* hidden_weights, float* hidden_biases, float* hidden_output,
                         float* output_weights, float* output_biases, float* q_values,
                         int batch_size, int state_size, int hidden_size, int action_size);
}

namespace game {
    // Game environment class (to be implemented)
    class GridWorld {
    public:
        GridWorld(int width, int height);
        int reset();
        std::tuple<int, float, bool> step(int action);
        std::vector<std::vector<int>> get_state_representation();
        int get_state_space_size() const;
        int get_action_space_size() const;
    };
}

PYBIND11_MODULE(cuda_gridworld_bindings, m) {
    m.doc() = "CUDA-accelerated reinforcement learning gridworld environment";
    
    // Expose AI functions
    m.def("update_q_table", [](py::array_t<float> q_table, 
                              py::array_t<int> states,
                              py::array_t<int> actions,
                              py::array_t<float> rewards,
                              py::array_t<int> next_states,
                              float learning_rate,
                              float discount_factor) {
        auto q_table_buf = q_table.request();
        auto states_buf = states.request();
        auto actions_buf = actions.request();
        auto rewards_buf = rewards.request();
        auto next_states_buf = next_states.request();
        
        int batch_size = states_buf.shape[0];
        int state_space_size = q_table_buf.shape[0];
        int action_space_size = q_table_buf.shape[1];
        
        ai::update_q_table_cuda(
            static_cast<float*>(q_table_buf.ptr),
            static_cast<int*>(states_buf.ptr),
            static_cast<int*>(actions_buf.ptr),
            static_cast<float*>(rewards_buf.ptr),
            static_cast<int*>(next_states_buf.ptr),
            learning_rate,
            discount_factor,
            batch_size,
            state_space_size,
            action_space_size
        );
    }, "Update Q-table using CUDA acceleration");
    
    m.def("get_best_action", [](py::array_t<float> q_table, int state_id) {
        auto q_table_buf = q_table.request();
        int action_space_size = q_table_buf.shape[1];
        
        return ai::get_best_action(
            static_cast<float*>(q_table_buf.ptr),
            state_id,
            action_space_size
        );
    }, "Get best action for a given state from Q-table");
    
    m.def("dqn_forward_cuda", [](py::array_t<float> states,
                               py::array_t<float> hidden_weights,
                               py::array_t<float> hidden_biases,
                               py::array_t<float> hidden_output,
                               py::array_t<float> output_weights,
                               py::array_t<float> output_biases,
                               py::array_t<float> q_values,
                               int batch_size,
                               int state_size,
                               int hidden_size,
                               int action_size) {
        auto states_buf = states.request();
        auto hidden_weights_buf = hidden_weights.request();
        auto hidden_biases_buf = hidden_biases.request();
        auto hidden_output_buf = hidden_output.request();
        auto output_weights_buf = output_weights.request();
        auto output_biases_buf = output_biases.request();
        auto q_values_buf = q_values.request();
        
        ai::dqn_forward_cuda(
            static_cast<float*>(states_buf.ptr),
            static_cast<float*>(hidden_weights_buf.ptr),
            static_cast<float*>(hidden_biases_buf.ptr),
            static_cast<float*>(hidden_output_buf.ptr),
            static_cast<float*>(output_weights_buf.ptr),
            static_cast<float*>(output_biases_buf.ptr),
            static_cast<float*>(q_values_buf.ptr),
            batch_size,
            state_size,
            hidden_size,
            action_size
        );
    }, "Perform DQN forward pass using CUDA acceleration");
    
    // Expose GridWorld environment
    py::class_<game::GridWorld>(m, "GridWorld")
        .def(py::init<int, int>())
        .def("reset", &game::GridWorld::reset)
        .def("step", &game::GridWorld::step)
        .def("get_state_representation", &game::GridWorld::get_state_representation)
        .def("get_state_space_size", &game::GridWorld::get_state_space_size)
        .def("get_action_space_size", &game::GridWorld::get_action_space_size);
}