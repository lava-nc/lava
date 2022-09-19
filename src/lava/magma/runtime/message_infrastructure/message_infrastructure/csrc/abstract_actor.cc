// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include "abstract_actor.h"

namespace message_infrastructure {

AbstractActor::AbstractActor() {
    this->ctl_status_shm_ = GetSharedMemManager().AllocChannelSharedMemory<RwSharedMemory>(
                        sizeof(ActorCtrlStatus));
    this->ctl_status_shm_->Start();
}

void AbstractActor::Control(ActorCmd cmd) {
    this->ctl_status_shm_->Handle([cmd](void* data){
        auto ctrl_status = reinterpret_cast<ActorCtrlStatus*>(data);
        ctrl_status->cmd = cmd;
    });
}

bool AbstractActor::HandleCmd() {
    bool stop = false;
    this->ctl_status_shm_->Handle([&stop](void* data){
        auto ctrl_status = reinterpret_cast<ActorCtrlStatus*>(data);
        if(ctrl_status->cmd == ActorCmd::CmdStop) {
            stop = true;
        }
    });
    return stop;
}

void AbstractActor::SetStatus(ActorStatus status) {
    this->ctl_status_shm_->Handle([status](void* data){
        auto ctrl_status = reinterpret_cast<ActorCtrlStatus*>(data);
        ctrl_status->status = status;
    });
}

int AbstractActor::GetStatus() {
    int status;
    this->ctl_status_shm_->Handle([&status](void* data){
        auto ctrl_status = reinterpret_cast<ActorCtrlStatus*>(data);
        status = static_cast<int>(ctrl_status->status);
    });
    return status;
}

} // namespace message_infrastructure