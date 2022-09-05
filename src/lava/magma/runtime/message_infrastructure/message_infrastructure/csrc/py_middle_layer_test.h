// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef PY_MIDDLE_LAYER_TEST_H_
#define PY_MIDDLE_LAYER_TEST_H_

#include "py_middle_layer.h"
#include "message_infrastructure_logging.h"
#include <stdlib.h>
#include <iostream>
namespace message_infrastructure {

class SimplePort {
 public:
  // SimplePort(char dtype, int *dim, int* stride) {
  //   metadata = new MetaData();
  //   metadata->dtype = dtype;
  //   metadata->dim = dim;
  //   metadata->stride = stride;
  //   metadata->mdata = NULL;
  // }
  int set_data(MetaData* metadata) {
    // need to check data
    this->metadata = metadata;
    return 0;
  }
  MetaData* get_data() {
    return this->metadata;
  }
  void transfer() {
    LAVA_LOG(LOG_MP, "transfering...\n");
    if (this->metadata == nullptr) {
      LAVA_LOG_ERR("metadata is null\n");
    }
    LAVA_LOG(LOG_MP, "dim[0]:%d\n", this->metadata->dimensions[0]);
    // if (metadata->type == 'i') {
    if (1) {
      int *data = (int*) this->metadata->mdata;
      for (int i=0; i<this->metadata->dimensions[0]*this->metadata->strides[0]/sizeof(int); i++) {
        data[i] ++;
      }
    }
    LAVA_LOG(LOG_MP, "transfer achieved\n");
  }
 private:
  MetaData *metadata=nullptr;
};

class ProxySimplePort {
 public:
  ProxySimplePort() {
    port = new SimplePort();
  }
  py::object* set_data(py::object *obj) {
    LAVA_LOG(LOG_MP, "set data\n");
    MetaData *metadata = new MetaData();
    //MDataFromObject(metadata, obj);
    this->obj = obj;
    //port->set_data(metadata);
    std::cout << (long)this->obj << std::endl;
    return this->obj;
  }
  py::object* get_data() {
    LAVA_LOG(LOG_MP, "get data\n");
    //MetaData *metadata = port->get_data();
    //MDataToObject(metadata, this->obj);
    std::cout << (long)this->obj << std::endl;
    return this->obj;
  }
  void transfer(){
    port->transfer();
  }
 private:
  py::object *obj;
  SimplePort *port;
};




}  // namespace message_infrastructure

#endif  // PY_MIDDLE_LAYER_H_