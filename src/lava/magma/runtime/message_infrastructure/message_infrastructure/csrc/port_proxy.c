// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <numpy/arrayobject.h>
#include <Python.h>
#include "port_proxy.h"
#include "message_infrastructure_logging.h"

int trick() {
    import_array();
    return 0;
}

const int tricky_var = trick();

MetaDataPtr SendPortProxy::MDataFromObject_(py::object* object) {
  PyObject *obj = object->ptr();
  LAVA_LOG(LOG_LAYER, "start MDataFromObject\n");
  if (!PyArray_Check(obj)) {
    LAVA_LOG_ERR("The Object is not array tp is %s\n", Py_TYPE(obj)->tp_name);
    exit(-1);
  }

  LAVA_LOG(LOG_LAYER, "check obj achieved\n");

  auto array = reinterpret_cast<PyArrayObject*> (obj);
  if (!PyArray_ISWRITEABLE(array)) {
    LAVA_LOG(LOG_LAYER, "The array is not writeable\n");
  }

  // var from numpy
  int32_t ndim = PyArray_NDIM(array);
  auto dims = PyArray_DIMS(array);
  auto strides = PyArray_STRIDES(array);
  void* data_ptr = PyArray_DATA(array);
  // auto dtype = PyArray_Type(array);  // no work
  auto dtype = array->descr->type_num;
  auto element_size_in_bytes = PyArray_ITEMSIZE(array);
  auto tsize = PyArray_SIZE(array);

  // set metadata
  MetaDataPtr metadata= std::make_shared<MetaData>();
  metadata->nd = ndim;
  for (int i = 0; i < ndim; i++) {
    metadata->dims[i] = dims[i];
    metadata->strides[i] = strides[i]/element_size_in_bytes;
    if (strides[i] % element_size_in_bytes != 0) {
      LAVA_LOG_ERR("numpy array stride not a multiple of element bytes\n");
    }
  }
  metadata->type = dtype;
  metadata->mdata = data_ptr;
  metadata->elsize = element_size_in_bytes;
  metadata->total_size = tsize;

  return metadata;
}

py::object RecvPortProxy::MDataToObject_(MetaDataPtr metadata) {
  std::vector<npy_intp> dims(metadata->nd);
  std::vector<npy_intp> strides(metadata->nd);

  for (int i = 0; i < metadata->nd; i++) {
    dims[i] = metadata->dims[i];
    strides[i] = metadata->strides[i] * metadata->elsize;
  }

  PyObject *array = PyArray_New(
    &PyArray_Type,
    metadata->nd,
    dims.data(),
    metadata->type,
    strides.data(),
    metadata->mdata,
    metadata->elsize,
    NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE,
    nullptr);

  if (!array)
    return py::cast(0);

  return py::reinterpret_borrow<py::object>(array);
}
 