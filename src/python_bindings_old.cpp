#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <memory>
#include "core/environment.h"

namespace py = pybind11;

// Helper class to provide gym-like interface for Python
class PyEnvironment {
private:
    std::shared_ptr<cudarl::Environment> env;
    
public:    PyEnvironment(int width = 10, int height = 10) {
        env = std::shared_ptr<cudarl::Environment>(new cudarl::Environment(0, width, height));
    }
    
    py::array_t<float> reset() {
        env->reset();
        return getObservation();
    }
    
    py::tuple step(int action) {
        env->step(action);
        auto obs = getObservation();
        float reward = env->getReward();
        bool done = env->isDone();
        py::dict info;
        info["agent_x"] = env->getAgentX();
        info["agent_y"] = env->getAgentY();
        return py::make_tuple(obs, reward, done, info);
    }
    
    py::array_t<float> getObservation() {
        std::vector<float> grid_data = env->getGrid();
        int height = env->getHeight();
        int width = env->getWidth();
        
        // Create numpy array from vector data
        auto result = py::array_t<float>(height * width);
        py::buffer_info buf = result.request();
        float* ptr = static_cast<float*>(buf.ptr);
        
        // Copy data
        std::copy(grid_data.begin(), grid_data.end(), ptr);
        
        // Reshape to 2D array
        result.resize({height, width});
        
        return result;
    }
    
    int getWidth() const { return env->getWidth(); }
    int getHeight() const { return env->getHeight(); }
    int getAgentX() const { return env->getAgentX(); }
    int getAgentY() const { return env->getAgentY(); }
    float getReward() const { return env->getReward(); }
    bool isDone() const { return env->isDone(); }
    py::tuple getAgentPosition() const {
        return py::make_tuple(env->getAgentX(), env->getAgentY());
    }
};

PYBIND11_MODULE(cudarl_core_python, m) {
    m.doc() = "CudaRL-Arena Python bindings - Direct CUDA Environment Interface";
    
    // Expose the PyEnvironment wrapper to Python as "Environment"
    py::class_<PyEnvironment>(m, "Environment")
        .def(py::init<int, int>(), py::arg("width") = 10, py::arg("height") = 10,
             "Initialize CUDA-accelerated environment")
        .def("reset", &PyEnvironment::reset,
             "Reset environment and return initial observation")
        .def("step", &PyEnvironment::step, py::arg("action"),
             "Take action and return (observation, reward, done, info)")
        .def("get_observation", &PyEnvironment::getObservation,
             "Get current environment observation")
        .def("get_width", &PyEnvironment::getWidth)
        .def("get_height", &PyEnvironment::getHeight)
        .def("get_agent_x", &PyEnvironment::getAgentX)
        .def("get_agent_y", &PyEnvironment::getAgentY)
        .def("get_reward", &PyEnvironment::getReward)
        .def("is_done", &PyEnvironment::isDone)
        .def("get_agent_position", &PyEnvironment::getAgentPosition);
    
    // Add version and capability information
    m.attr("__version__") = "0.1.0";
    m.attr("cuda_available") = true;
}
