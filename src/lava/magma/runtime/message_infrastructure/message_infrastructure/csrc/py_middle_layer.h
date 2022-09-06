// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef PY_MIDDLE_LAYER_H_
#define PY_MIDDLE_LAYER_H_

#include <Python.h>
#include <pybind11/pybind11.h>
#include <numpy/arrayobject.h>

namespace message_infrastructure {

namespace py = pybind11;

struct MetaData {
  int nd;
  int64_t *dimensions;
  int64_t *strides;
  size_t size;
  char* mdata;
  char type;
};

py::object MDataToObject(MetaData*, py::object*);

void MDataFromObject(MetaData*, py::object*);

}  // namespace message_infrastructure

#endif  // PY_MIDDLE_LAYER_H_
