#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "environment_bridge.h"

namespace py = pybind11;

PYBIND11_MODULE(cudarl_core, m) {
    m.doc() = "CudaRL-Arena Python bindings";
    
    // Expose the EnvironmentBridge class to Python
    py::class_<EnvironmentBridge>(m, "Environment")
        .def(py::init<int, int>(), py::arg("width") = 10, py::arg("height") = 10)
        .def("reset", &EnvironmentBridge::reset)
        .def("step", &EnvironmentBridge::step)
        .def("get_observation", &EnvironmentBridge::getObservation)
        .def("get_width", &EnvironmentBridge::getWidth)
        .def("get_height", &EnvironmentBridge::getHeight)
        .def("get_agent_position", &EnvironmentBridge::getAgentPosition)
        .def("add_obstacle", &EnvironmentBridge::addObstacle)
        .def("set_goal", &EnvironmentBridge::setGoal)
        .def("add_trap", &EnvironmentBridge::addTrap);
    
    // Add version information
    m.attr("__version__") = "0.1.0";
}