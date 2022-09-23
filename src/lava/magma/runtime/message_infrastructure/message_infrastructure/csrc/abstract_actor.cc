// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include "abstract_actor.h"
#include <xmmintrin.h>

namespace message_infrastructure {

AbstractActor::AbstractActor(AbstractActor::TargetFn target_fn)
    : target_fn_(target_fn)
{
    this->ctl_status_shm_ = GetSharedMemManager().AllocChannelSharedMemory<RwSharedMemory>(
                        sizeof(ActorCtrlStatus));
    this->ctl_status_shm_->Start();
}

void AbstractActor::Control(const ActorCmd cmd) {
    this->ctl_status_shm_->Handle([cmd](void* data){
        auto ctrl_status = reinterpret_cast<ActorCtrlStatus*>(data);
        ctrl_status->cmd = cmd;
    });
}

bool AbstractActor::HandleStatus() {
    bool stop = false;
    int status = GetStatus();
    switch (status){
        case ActorStatus::StatusError:
        case ActorStatus::StatusStopped:
            stop = true;
            stop_fn_();
            break;
        default:
            break;
    }
    return stop;
}

std::pair<bool, bool> AbstractActor::HandleCmd() {
    bool stop = false;
    bool wait = false;
    this->ctl_status_shm_->Handle([&stop, &wait](void* data){
        auto ctrl_status = reinterpret_cast<ActorCtrlStatus*>(data);
        if(ctrl_status->cmd == ActorCmd::CmdStop) {
            stop = true;
        }
        else if (ctrl_status->cmd == ActorCmd::CmdPause)
        {
            ctrl_status->status = ActorStatus::StatusPaused;
            wait = true;
        }
        else if (ctrl_status->cmd == ActorCmd::CmdRun)
        {
            ctrl_status->status = ActorStatus::StatusRunning;
        }
    });
    return std::make_pair(stop, wait);
}

void AbstractActor::SetStatus(ActorStatus status) {
    this->ctl_status_shm_->Handle([status](void* data){
        auto ctrl_status = reinterpret_cast<ActorCtrlStatus*>(data);
        ctrl_status->status = status;
    });
}

int AbstractActor::SetStopFn(StopFn stop_fn) {
    stop_fn_ = stop_fn;
    if(stop_fn_ == nullptr) {
        LAVA_LOG_ERR("Set None Function\n");
        return -1;
    }
    actor_monitor_ = std::make_shared<std::thread>(&message_infrastructure::AbstractActor::ActorMonitor_, this);
    return 0;
}

int AbstractActor::ActorMonitor_() {
    while(true) {
        if(HandleStatus())
            break;
        HandleCmd();
    }
    return 0;
}

int AbstractActor::GetStatus() {
    int status;
    this->ctl_status_shm_->Handle([&status](void* data){
        auto ctrl_status = reinterpret_cast<ActorCtrlStatus*>(data);
        status = static_cast<int>(ctrl_status->status);
    });
    return status;
}

int AbstractActor::GetCmd() {
    int cmd;
    this->ctl_status_shm_->Handle([&cmd](void* data){
        auto ctrl_status = reinterpret_cast<ActorCtrlStatus*>(data);
        cmd = static_cast<int>(ctrl_status->cmd);
    });
    return cmd;
}

void AbstractActor::Run() {
    InitStatus();
    while(true) {
      auto handle = HandleCmd();
      if (handle.first) {
        break;
      }
      if (!handle.second) {
        target_fn_(this);
        LAVA_LOG(LOG_MP, "Actor: ActorStatus:%d\n", GetStatus());
      } else {
        // waiting
        _mm_pause();
      }
    }
    SetStatus(ActorStatus::StatusStopped);
    actor_monitor_->join();
    LAVA_LOG(LOG_ACTOR, "child exist, pid:%d\n", this->pid_);
}

void AbstractActor::InitStatus() {
    this->ctl_status_shm_->Handle([](void* data){
        auto ctrl_status = reinterpret_cast<ActorCtrlStatus*>(data);
        ctrl_status->status = ActorStatus::StatusRunning;
        ctrl_status->cmd = ActorCmd::CmdRun;
    });
}

} // namespace message_infrastructure