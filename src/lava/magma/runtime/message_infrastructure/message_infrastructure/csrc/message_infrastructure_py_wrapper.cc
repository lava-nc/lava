// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <memory>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "abstract_actor.h"
#include "channel_proxy.h"
#include "multiprocessing.h"
#include "port_proxy.h"
#include "utils.h"

namespace message_infrastructure {

namespace py = pybind11;

PYBIND11_MODULE(MessageInfrastructurePywrapper, m) {
  py::class_<MultiProcessing> (m, "CppMultiProcessing")
    .def(py::init<>())
    .def("build_actor", &MultiProcessing::BuildActor)
    .def("check_actor", &MultiProcessing::CheckActor)
    .def("get_actors", &MultiProcessing::GetActors, py::return_value_policy::reference)
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
    .def_property_readonly("src_port", &ChannelProxy::GetSendPort, py::return_value_policy::reference)
    .def_property_readonly("dst_port", &ChannelProxy::GetRecvPort, py::return_value_policy::reference);
  py::class_<SendPortProxy, PortProxy, std::shared_ptr<SendPortProxy>> (m, "SendPort")
    .def(py::init<>())
    .def("get_channel_type", &SendPortProxy::GetChannelType)
    .def("start", &SendPortProxy::Start)
    .def("probe", &SendPortProxy::Probe)
    .def("send", &SendPortProxy::Send)
    .def("join", &SendPortProxy::Join)
    .def_property_readonly("name", &SendPortProxy::Name)
    .def("size", &SendPortProxy::Size);
  py::class_<RecvPortProxy, PortProxy, std::shared_ptr<RecvPortProxy>> (m, "RecvPort")
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
