#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

#include "multiprocessing.h"
//#include "abstract_port_implementation.h"

namespace py = pybind11;

PYBIND11_MODULE(PyWrapper, m) {
  py::class_<MultiProcessing> (m, "MultiProcessing")
    .def(py::init<>())
    .def("build_actor", &MultiProcessing::BuildActor)
    .def("check_actor", &MultiProcessing::CheckActor)
    .def("stop", &MultiProcessing::Stop);
}
