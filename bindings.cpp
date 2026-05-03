#include <pybind11/pybind11.h>
#include <pybind11/stl.h>      // handles vector conversion automatically
#include <pybind11/numpy.h>    // numpy buffer protocol
#include "rotorQuant.h"

namespace py = pybind11;

PYBIND11_MODULE(rotorquant, m){
    py::class_<RotorQuant>(m, "RotorQuant")
        .def(py::init<int, int, double>())
        .def("encode", &RotorQuant::encode)
        .def("decode", &RotorQuant::decode)
        .def("encode_2d", &RotorQuant::encode_2d)
        .def("decode_2d", &RotorQuant::decode_2d)
        .def("encode_decode_2d_inplace",
             [](RotorQuant& self,
                py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
                 auto buf = arr.request();
                 if (buf.ndim != 2)
                     throw std::runtime_error("expected 2D float64 array");
                 py::gil_scoped_release release;
                 self.encode_decode_2d_inplace(
                     static_cast<double*>(buf.ptr),
                     static_cast<int>(buf.shape[0]),
                     static_cast<int>(buf.shape[1]));
             })
        .def("encode_decode_batch_f32",
             [](RotorQuant& self,
                py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
                 auto buf = arr.request();
                 if (buf.ndim != 2)
                     throw std::runtime_error("expected 2D float32 array");
                 py::gil_scoped_release release;
                 self.encode_decode_batch_f32(
                     static_cast<float*>(buf.ptr),
                     static_cast<int>(buf.shape[0]),
                     static_cast<int>(buf.shape[1]));
             })
        .def("flips_f32",  [](RotorQuant& self){ return self.flips_f32_; })
        .def("bp_f32",     [](RotorQuant& self){ return self.breakpoints_f32_; })
        .def("cent_f32",   [](RotorQuant& self){ return self.centroids_f32_; })
        .def("num_levels", [](RotorQuant& self){ return (int)self.centroids_f32_.size(); });
}
