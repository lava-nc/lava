// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <message_infrastructure/csrc/core/abstract_actor.h>
#include <message_infrastructure/csrc/core/multiprocessing.h>
#include <message_infrastructure/csrc/port_proxy.h>
#include <message_infrastructure/csrc/core/utils.h>
#include <message_infrastructure/csrc/channel_proxy.h>
#include <message_infrastructure/csrc/actor/posix_actor.h>

#include <memory>

namespace message_infrastructure {

namespace py = pybind11;

// Users should be allowed to copy channel objects.
// Use std::shared_ptr.
#if defined(GRPC_CHANNEL)
using GetRPCChannelProxyPtr = std::shared_ptr<GetRPCChannelProxy>;
#endif

#if defined(DDS_CHANNEL)
using GetDDSChannelProxyPtr = std::shared_ptr<GetDDSChannelProxy>;
#endif

PYBIND11_MODULE(MessageInfrastructurePywrapper, m) {
  py::class_<MultiProcessing> (m, "CppMultiProcessing")
    .def(py::init<>())
    .def("build_actor", &MultiProcessing::BuildActor)
    .def("check_actor", &MultiProcessing::CheckActor)
    .def("get_actors", &MultiProcessing::GetActors,
                       py::return_value_policy::reference)
    .def("cleanup", &MultiProcessing::Cleanup)
    .def("stop", &MultiProcessing::Stop);
  py::enum_<ProcessType> (m, "ProcessType")
    .value("ErrorProcess", ErrorProcess)
    .value("ChildProcess", ChildProcess)
    .value("ParentProcess", ParentProcess)
    .export_values();
  py::enum_<ActorStatus> (m, "ActorStatus")
    .value("StatusError", ActorStatus::StatusError)
    .value("StatusRunning", ActorStatus::StatusRunning)
    .value("StatusStopped", ActorStatus::StatusStopped)
    .value("StatusPaused", ActorStatus::StatusPaused)
    .value("StatusTerminated", ActorStatus::StatusTerminated)
    .export_values();
  py::enum_<ActorCmd> (m, "ActorCmd")
    .value("CmdRun", ActorCmd::CmdRun)
    .value("CmdStop", ActorCmd::CmdStop)
    .value("CmdPause", ActorCmd::CmdPause)
    .export_values();
  py::class_<PosixActor> (m, "Actor")
    .def("wait", &PosixActor::Wait)
    .def("get_status", &PosixActor::GetStatus)
    .def("set_stop_fn", &PosixActor::SetStopFn)
    .def("pause", [](PosixActor &actor){
        actor.Control(ActorCmd::CmdPause);
      })
    .def("start", [](PosixActor &actor){
        actor.Control(ActorCmd::CmdRun);
      })
    .def("stop", [](PosixActor &actor){
        actor.Control(ActorCmd::CmdStop);
      })
    .def("status_stopped", [](PosixActor &actor){
        return actor.SetStatus(ActorStatus::StatusStopped);
      })
    .def("status_running", [](PosixActor &actor){
        return actor.SetStatus(ActorStatus::StatusRunning);
      })
    .def("status_paused", [](PosixActor &actor){
        return actor.SetStatus(ActorStatus::StatusPaused);
      })
    .def("status_terminated", [](PosixActor &actor){
        return actor.SetStatus(ActorStatus::StatusTerminated);
      })
    .def("error", [](PosixActor &actor){
        return actor.SetStatus(ActorStatus::StatusError);
      });
  py::enum_<ChannelType> (m, "ChannelType")
    .value("SHMEMCHANNEL", SHMEMCHANNEL)
    .value("RPCCHANNEL", RPCCHANNEL)
    .value("DDSCHANNEL", DDSCHANNEL)
    .value("SOCKETCHANNEL", SOCKETCHANNEL)
    .export_values();
  py::class_<PortProxy, std::shared_ptr<PortProxy>> (m, "AbstractTransferPort")
    .def(py::init<>());
  py::class_<ChannelProxy, std::shared_ptr<ChannelProxy>> (m, "Channel")
    .def(py::init<ChannelType, size_t, size_t, std::string, std::string>())
    .def_property_readonly("src_port", &ChannelProxy::GetSendPort,
                                       py::return_value_policy::reference)
    .def_property_readonly("dst_port", &ChannelProxy::GetRecvPort,
                                       py::return_value_policy::reference);
#if defined(GRPC_CHANNEL)
  py::class_<GetRPCChannelProxy, GetRPCChannelProxyPtr> (m, "GetRPCChannel")
    .def(py::init<std::string, int, std::string, std::string, size_t>())
    .def(py::init<std::string, std::string, size_t>())
    .def_property_readonly("src_port", &GetRPCChannelProxy::GetSendPort,
                                       py::return_value_policy::reference)
    .def_property_readonly("dst_port", &GetRPCChannelProxy::GetRecvPort,
                                       py::return_value_policy::reference);
#endif
  m.def("support_grpc_channel", [](){
#if defined(GRPC_CHANNEL)
    return true;
#else
    return false;
#endif
  });

#if defined(DDS_CHANNEL)
  py::enum_<DDSTransportType> (m, "DDSTransportType")
    .value("DDSSHM", DDSSHM)
    .value("DDSTCPv4", DDSTCPv4)
    .value("DDSTCPv6", DDSTCPv6)
    .value("DDSUDPv4", DDSUDPv4)
    .value("DDSUDPv6", DDSUDPv6)
    .export_values();

  py::enum_<DDSBackendType> (m, "DDSBackendType")
    .value("FASTDDSBackend", FASTDDSBackend)
    .value("CycloneDDSBackend", CycloneDDSBackend)
    .export_values();

  py::class_<GetDDSChannelProxy, GetDDSChannelProxyPtr> (m, "GetDDSChannel")
    .def(py::init<std::string, DDSTransportType, DDSBackendType, size_t>())
    .def_property_readonly("src_port", &GetDDSChannelProxy::GetSendPort,
                                       py::return_value_policy::reference)
    .def_property_readonly("dst_port", &GetDDSChannelProxy::GetRecvPort,
                                       py::return_value_policy::reference);
#endif

  m.def("support_fastdds_channel", [](){
#if defined(FASTDDS_ENABLE)
    return true;
#else
    return false;
#endif
  });

  m.def("support_cyclonedds_channel", [](){
#if defined(CycloneDDS_ENABLE)
    return true;
#else
    return false;
#endif
  });

  py::class_<SendPortProxy, PortProxy,
             std::shared_ptr<SendPortProxy>> (m, "SendPort")
    .def(py::init<>())
    .def("get_channel_type", &SendPortProxy::GetChannelType)
    .def("start", &SendPortProxy::Start)
    .def("probe", &SendPortProxy::Probe)
    .def("send", &SendPortProxy::Send)
    .def("join", &SendPortProxy::Join)
    .def_property_readonly("name", &SendPortProxy::Name)
    .def("size", &SendPortProxy::Size);
  py::class_<RecvPortProxy, PortProxy,
             std::shared_ptr<RecvPortProxy>> (m, "RecvPort")
    .def(py::init<>())
    .def("get_channel_type", &RecvPortProxy::GetChannelType)
    .def("start", &RecvPortProxy::Start)
    .def("probe", &RecvPortProxy::Probe)
    .def("recv", &RecvPortProxy::Recv)
    .def("peek", &RecvPortProxy::Peek)
    .def("join", &RecvPortProxy::Join)
    .def_property_readonly("name", &RecvPortProxy::Name)
    .def("size", &RecvPortProxy::Size);
}

}  // namespace message_infrastructure
