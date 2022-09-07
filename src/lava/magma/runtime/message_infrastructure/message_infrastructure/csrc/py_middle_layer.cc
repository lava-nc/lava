// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include "py_middle_layer.h"
#include "message_infrastructure_logging.h"

namespace message_infrastructure {

int numpy_trick() {
  import_array();
  return 1;
}

const int NumpyTrick = numpy_trick();

size_t PyArraySize(PyArrayObject* array) {
  return array->dimensions[0] * array->strides[0];
}

py::object MDataToObject(MetaData* meta_data, py::object* object) {
  LAVA_LOG(LOG_MP, "transfer mdata to object\n");
  PyObject *ptr = object->ptr();
  if (!PyArray_Check(ptr)) {
    LAVA_LOG_ERR("The Object is not array tp is %s\n", Py_TYPE(ptr)->tp_name);
    exit(-1);
  }
  
  auto array = (PyArrayObject*)ptr;
  array->data = meta_data->mdata;
  LAVA_LOG(LOG_MP, "transfer mdata to object achieved\n");
  return  py::reinterpret_borrow<py::object> (ptr);
}

void MDataFromObject(MetaData* meta_data, py::object* object) {
  LAVA_LOG(LOG_MP, "transfer mdata from object\n");
  PyObject *ptr = object->ptr();
  if (!PyArray_Check(ptr)) {
    LAVA_LOG_ERR("The Object is not array tp is %s\n", Py_TYPE(ptr)->tp_name);
    return;
  }

  auto array = (PyArrayObject*)ptr;
  meta_data->nd = array->nd;
  meta_data->dimensions = array->dimensions;
  meta_data->strides = array->strides;
  meta_data->mdata = array->data;
  meta_data->size = PyArraySize(array);
  LAVA_LOG(LOG_MP, "transfer mdata from object achieved, dim[0]:%ld, stride[0]:%ld\n", meta_data->dimensions[0], array->strides[0]);
}

}  // namespace message_infrastructure
