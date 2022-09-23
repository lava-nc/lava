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
    LAVA_DEBUG(LOG_ACTOR, "Actor%d add Control %d\n",pid_, cmd);
    this->ctl_status_shm_->Handle([cmd](void* data){
        auto ctrl_status = reinterpret_cast<ActorCtrlStatus*>(data);
        ctrl_status->cmd = cmd;
    });
    LAVA_DEBUG(LOG_ACTOR, "Actor%d added Control %d\n",pid_, cmd);
}

std::pair<bool, bool> AbstractActor::HandleStatus() {
    bool stop = false;
    bool wait = false;
    int status = GetStatus();
    if (status != ctl_status_backup.status) {
        LAVA_DEBUG(LOG_ACTOR, "actor %d status changed from %d to %d\n", pid_, ctl_status_backup.status, status);
        ctl_status_backup.status = static_cast<ActorStatus>(status);
    }
    switch (status){
        case ActorStatus::StatusError:
            LAVA_DEBUG(LOG_ACTOR, "actor %d status error\n", pid_);
        case ActorStatus::StatusStopped:
            stop = true;
            LAVA_DEBUG(LOG_ACTOR, "actor %d trigger stop fun\n", pid_);
            stop_fn_();
            LAVA_DEBUG(LOG_ACTOR, "actor %d triggered stop fun\n", pid_);
            break;
        case ActorStatus::StatusPaused:
            wait = true;
        default:
            break;
    }
    return std::make_pair(stop, wait);
}

void AbstractActor::HandleCmd() {
    this->ctl_status_shm_->Handle([](void* data){
        auto ctrl_status = reinterpret_cast<ActorCtrlStatus*>(data);
        if(ctrl_status->cmd == ActorCmd::CmdStop) {
            ctrl_status->status = ActorStatus::StatusStopped;
        }
        else if (ctrl_status->cmd == ActorCmd::CmdPause)
        {
            ctrl_status->status = ActorStatus::StatusPaused;
        }
        else if (ctrl_status->cmd == ActorCmd::CmdRun)
        {
            ctrl_status->status = ActorStatus::StatusRunning;
        }
    });
}

void AbstractActor::SetStatus(ActorStatus status) {
    this->ctl_status_shm_->Handle([status](void* data){
        auto ctrl_status = reinterpret_cast<ActorCtrlStatus*>(data);
        ctrl_status->status = status;
    });
}

int AbstractActor::SetStopFn(StopFn stop_fn) {
    stop_fn_ = stop_fn;
    LAVA_DEBUG(LOG_ACTOR, "set actor stop function for pid:%d\n", pid_);
    if(stop_fn_ == nullptr) {
        LAVA_LOG_ERR("Set None Function\n");
        return -1;
    }
    actor_monitor_ = std::make_shared<std::thread>(&message_infrastructure::AbstractActor::ActorMonitor_, this);
    LAVA_DEBUG(LOG_ACTOR, "Create thread polling for pid:%d\n", pid_);
    return 0;
}

void AbstractActor::ActorMonitor_() {
    int i=0;
    while(true) {
        i++;
        auto handle = HandleStatus();
        if(handle.first)
            break;
        HandleCmd();
        LAVA_DEBUG(LOG_ACTOR, "I have run %d times\n", i);
    }
    LAVA_DEBUG(LOG_ACTOR, "I have run %d times\n", i);
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
      LAVA_DEBUG(LOG_ACTOR, "Actor %d Run\n", pid_);
      target_fn_(this);
      LAVA_LOG(LOG_ACTOR, "Actor: ActorStatus:%d, pid:%d\n", GetStatus(), this->pid_);
      auto handle = HandleStatus();
      if (handle.first) {
        break;
      }
      if (!handle.second) {
        LAVA_DEBUG(LOG_ACTOR, "Handle the actor\n");
        continue;
      } else {
        // waiting
        LAVA_DEBUG(LOG_ACTOR, "Actor %d pause\n", pid_);
        _mm_pause();
      }
    }
    SetStatus(ActorStatus::StatusStopped);
    LAVA_DEBUG(LOG_ACTOR, "Join\n");
    LAVA_LOG(LOG_ACTOR, "child exist, pid:%d\n", this->pid_);
}

void AbstractActor::InitStatus() {
    this->ctl_status_shm_->Handle([](void* data){
        auto ctrl_status = reinterpret_cast<ActorCtrlStatus*>(data);
        ctrl_status->status = ActorStatus::StatusPaused;
        ctrl_status->cmd = ActorCmd::CmdPause;
    });
    this->ctl_status_backup.cmd = ActorCmd::CmdPause;
    this->ctl_status_backup.status = ActorStatus::StatusPaused;
}

} // namespace message_infrastructure