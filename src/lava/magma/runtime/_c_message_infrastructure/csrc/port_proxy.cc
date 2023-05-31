// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#define NUMPY_CORE_INCLUDE_NUMPY_NPY_1_7_DEPRECATED_API_H_
// to solve the warning "Using deprecated NumPy API,
// disable it with " "#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION"

#include <numpy/arrayobject.h>
#include <Python.h>
#include <port_proxy.h>
#include <core/message_infrastructure_logging.h>

namespace message_infrastructure {

namespace py = pybind11;

#if defined(GRPC_CHANNEL)
DataPtr GrpcMDataFromObject_(py::object* object) {
  PyObject *obj = object->ptr();
  LAVA_LOG(LOG_LAYER, "start GrpcMDataFromObject\n");
  if (!PyArray_Check(obj)) {
    LAVA_LOG_FATAL("The Object is not array tp is %s\n", Py_TYPE(obj)->tp_name);
    exit(-1);
  }
  LAVA_LOG(LOG_LAYER, "check obj achieved\n");
  auto array = reinterpret_cast<PyArrayObject*> (obj);
  if (!PyArray_ISWRITEABLE(array)) {
    LAVA_LOG(LOG_LAYER, "The array is not writeable\n");
  }
  int32_t ndim = PyArray_NDIM(array);
  auto dims = PyArray_DIMS(array);
  auto strides = PyArray_STRIDES(array);
  void* data_ptr = PyArray_DATA(array);
  auto dtype = array->descr->type_num;
  auto element_size_in_bytes = PyArray_ITEMSIZE(array);
  auto tsize = PyArray_SIZE(array);
  // set grpcdata
  GrpcMetaDataPtr grpcdata = std::make_shared<GrpcMetaData>();
  grpcdata->set_nd(ndim);
  grpcdata->set_type(dtype);
  grpcdata->set_elsize(element_size_in_bytes);
  grpcdata->set_total_size(tsize);
  for (int i = 0; i < ndim; i++) {
    grpcdata->add_dims(dims[i]);
    grpcdata->add_strides(strides[i]/element_size_in_bytes);
    if (strides[i] % element_size_in_bytes != 0) {
      LAVA_LOG_FATAL("Numpy array stride not a multiple of element bytes\n");
    }
  }
  char* data = reinterpret_cast<char*>(data_ptr);
  grpcdata->set_value(data, element_size_in_bytes*tsize);
  return grpcdata;
}
#endif

DataPtr MDataFromObject_(py::object* object) {
  PyObject *obj = object->ptr();
  LAVA_LOG(LOG_LAYER, "start MDataFromObject\n");
  if (!PyArray_Check(obj)) {
    LAVA_LOG_FATAL("The Object is not array tp is %s\n", Py_TYPE(obj)->tp_name);
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
  MetaDataPtr metadata = std::make_shared<MetaData>();
  metadata->nd = ndim;
  for (int i = 0; i < ndim; i++) {
    metadata->dims[i] = dims[i];
    metadata->strides[i] = strides[i]/element_size_in_bytes;
    if (strides[i] % element_size_in_bytes != 0) {
      LAVA_LOG_ERR("Numpy array stride not a multiple of element bytes\n");
    }
  }
  metadata->type = dtype;
  metadata->mdata = data_ptr;
  metadata->elsize = element_size_in_bytes;
  metadata->total_size = tsize;
  return metadata;
}

void MetaDataDump(MetaDataPtr metadata) {
  int64_t *dims = metadata->dims;
  int64_t *strides = metadata->strides;
  LAVA_DUMP(LOG_LAYER, "MetaData Info:\n"
               "(nd, type, elsize): (%ld, %ld, %ld)\n"
               "total_size: %ld\n"
               "dims:[%ld, %ld, %ld, %ld, %ld]\n"
               "strides:[%ld, %ld, %ld, %ld, %ld]\n",
               metadata->nd,
               metadata->type,
               metadata->elsize,
               metadata->total_size,
               dims[0], dims[1], dims[2], dims[3], dims[4],
               strides[0], strides[1], strides[2], strides[3], strides[4]);
}

py::object PortProxy::DType() {
  return d_type_;
}

py::tuple PortProxy::Shape() {
  return shape_;
}

ChannelType SendPortProxy::GetChannelType() {
  return channel_type_;
}
void SendPortProxy::Start() {
  send_port_->Start();
}
bool SendPortProxy::Probe() {
  return send_port_->Probe();
}
void SendPortProxy::Send(py::object* object) {
  DataPtr data = DataFromObject_(object);
  send_port_->Send(data);
}
void SendPortProxy::Join() {
  send_port_->Join();
}
std::string SendPortProxy::Name() {
  return send_port_->Name();
}
size_t SendPortProxy::Size() {
  return send_port_->Size();
}

ChannelType RecvPortProxy::GetChannelType() {
  return channel_type_;
}
void RecvPortProxy::Start() {
  recv_port_->Start();
}
bool RecvPortProxy::Probe() {
  return recv_port_->Probe();
}
py::object RecvPortProxy::Recv() {
  MetaDataPtr metadata = recv_port_->Recv();
  return MDataToObject_(metadata);
}
void RecvPortProxy::Join() {
  recv_port_->Join();
}
py::object RecvPortProxy::Peek() {
  MetaDataPtr metadata = recv_port_->Peek();
  return MDataToObject_(metadata);
}
std::string RecvPortProxy::Name() {
  return recv_port_->Name();
}
size_t RecvPortProxy::Size() {
  return recv_port_->Size();
}

void RecvPortProxy::SetObserver(std::function<void()> obs) {
  recv_port_->obs_lk_.lock();
  if (obs)
    recv_port_->observer_ = obs;
  else
    recv_port_->observer_ = nullptr;
  recv_port_->obs_lk_.unlock();
}

int trick() {
    // to solve the warning "converting to non-pointer type 'int'
    // from NULL [-Wconversion-null] import_array()"
    _import_array();
    return 0;
}

const int tricky_var = trick();

DataPtr SendPortProxy::DataFromObject_(py::object* object) {
#if defined(GRPC_CHANNEL)
  if (channel_type_== ChannelType::RPCCHANNEL) {
    return GrpcMDataFromObject_(object);
  }
#endif
  return MDataFromObject_(object);
}

py::object RecvPortProxy::MDataToObject_(MetaDataPtr metadata) {
  if (metadata == nullptr)
    return py::cast(0);

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
  LAVA_DEBUG(LOG_LAYER, "Set PyObject capsule, mdata: %p\n", metadata->mdata);
  PyObject *capsule = PyCapsule_New(metadata->mdata, nullptr,
                                      [](PyObject *capsule) {
    void *memory = PyCapsule_GetPointer(capsule, nullptr);
    LAVA_DEBUG(LOG_LAYER, "PyObject cleaned, free memory: %p.\n", memory);
    free(memory);
    LAVA_DEBUG(LOG_LAYER, "memory has been released\n");});
  LAVA_ASSERT_INT(nullptr == capsule, 0);
  LAVA_ASSERT_INT(PyArray_SetBaseObject(
                  reinterpret_cast<PyArrayObject *>(array),
                  capsule), 0);
  return py::reinterpret_steal<py::object>(array);
}


void Selector::Changed() {
  std::unique_lock<std::mutex> lock(cv_mutex_);
  ready_ = true;
  cv_.notify_all();
}

void Selector::SetObserver(std::vector<std::tuple<RecvPortProxyPtr,
                           py::function>> *channel_actions,
                           std::function<void()> observer) {
  for (auto it = channel_actions->begin();
      it != channel_actions->end(); ++it) {
    std::get<0>(*it)->SetObserver(observer);
  }
}

pybind11::object Selector::Select(std::vector<std::tuple<RecvPortProxyPtr,
                                  py::function>> *args) {
  std::function<void()> observer = std::bind(&Selector::Changed, this);
  SetObserver(args, observer);
  while (true) {
    for (auto it = args->begin(); it != args->end(); ++it) {
      if (std::get<0>(*it)->Probe()) {
        SetObserver(args, nullptr);
        return std::get<1>(*it)();
      }
    }
    std::unique_lock<std::mutex> lock(cv_mutex_);
    cv_.wait(lock, [this]{return ready_;});
    ready_ = false;
  }
}

}  // namespace message_infrastructure
