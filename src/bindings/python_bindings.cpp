#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "../core/environment.h"

namespace py = pybind11;
using namespace cudarl;

PYBIND11_MODULE(cudarl_core, m) {
    m.doc() = "CudaRL-Arena Python bindings for CUDA-accelerated reinforcement learning";
    
    // Version info
    m.attr("__version__") = "0.1.0";
    
    // Environment class
    py::class_<Environment>(m, "Environment")
        .def(py::init<int, int, int>(),
             py::arg("env_id") = 0,
             py::arg("width") = 10,
             py::arg("height") = 10)
        .def("reset", &Environment::reset)
        .def("step", &Environment::step)
        .def("get_width", &Environment::getWidth)
        .def("get_height", &Environment::getHeight)
        .def("get_agent_x", &Environment::getAgentX)
        .def("get_agent_y", &Environment::getAgentY)
        .def("get_reward", &Environment::getReward)
        .def("is_done", &Environment::isDone)
        .def("get_cell_value", &Environment::getCellValue)
        .def("get_grid_data", [](const Environment& env) {
            std::vector<float> grid = env.getGrid();
            const size_t size = grid.size();
            
            // Create a NumPy array from the grid data
            auto result = py::array_t<float>(size);
            py::buffer_info buf = result.request();
            float* ptr = static_cast<float*>(buf.ptr);
            
            // Copy data
            std::memcpy(ptr, grid.data(), size * sizeof(float));
            
            return result;
        });
}