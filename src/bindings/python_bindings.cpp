#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "../environment.h"

namespace py = pybind11;

// Wrapper class for the C++ Environment
class PyEnvironment {
public:
    PyEnvironment(int width, int height, int env_id = 0)
        : width(width), height(height), env_id(env_id), env(env_id, width, height) {}
    
    py::array_t<float> reset() {
        env.reset();
        return getObservation();
    }
    
    std::tuple<py::array_t<float>, float, bool, py::dict> step(int action) {
        env.step(action);
        
        // Get observation
        auto obs = getObservation();
        
        // Get reward and done flag
        float reward = env.getReward();
        bool done = env.isDone();
        
        // Create info dictionary
        py::dict info;
        info["agent_x"] = env.getAgentX();
        info["agent_y"] = env.getAgentY();
        
        return std::make_tuple(obs, reward, done, info);
    }
    
    py::array_t<float> getObservation() {
        // Get grid data from environment
        std::vector<float> grid = env.getGrid();
        
        // Create numpy array from grid data
        auto obs = py::array_t<float>({height, width});
        py::buffer_info buf = obs.request();
        float* ptr = static_cast<float*>(buf.ptr);
        
        // Copy grid data to numpy array (reshape from 1D to 2D)
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                ptr[y * width + x] = grid[y * width + x];
            }
        }
        
        return obs;
    }
    
    int getWidth() const { return width; }
    int getHeight() const { return height; }
    int getEnvId() const { return env_id; }
    
private:
    int width;
    int height;
    int env_id;
    cudarl::Environment env;
};

PYBIND11_MODULE(cudarl_core, m) {
    m.doc() = "CudaRL-Arena Python bindings";
    
    py::class_<PyEnvironment>(m, "Environment")
        .def(py::init<int, int, int>(), 
             py::arg("width") = 10, 
             py::arg("height") = 10, 
             py::arg("env_id") = 0)
        .def("reset", &PyEnvironment::reset)
        .def("step", &PyEnvironment::step)
        .def("get_observation", &PyEnvironment::getObservation)
        .def_property_readonly("width", &PyEnvironment::getWidth)
        .def_property_readonly("height", &PyEnvironment::getHeight)
        .def_property_readonly("env_id", &PyEnvironment::getEnvId);
}