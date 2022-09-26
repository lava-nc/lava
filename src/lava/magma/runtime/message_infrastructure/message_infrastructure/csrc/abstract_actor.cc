// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include "abstract_actor.h"
#include <xmmintrin.h>

namespace message_infrastructure {

AbstractActor::AbstractActor(AbstractActor::TargetFn target_fn)
    : target_fn_(target_fn)
{
    this->ctl_shm_ = GetSharedMemManager().AllocChannelSharedMemory<SharedMemory>(
                        sizeof(int));
    this->ctl_shm_->Start();
}

AbstractActor::~AbstractActor() {
  ctl_shm_->Close();
}

void AbstractActor::Control(const ActorCmd cmd) {
    this->ctl_shm_->Store([cmd](void* data){
        auto ctrl_cmd = reinterpret_cast<int *>(data);
        *ctrl_cmd = static_cast<int>(cmd);
    });
}

void AbstractActor::HandleCmd() {
    while(actore_status_.load() < static_cast<int>(ActorStatus::StatusStopped)) {
        auto ret = ctl_shm_->Load([this](void *data){
            auto ctrl_status = reinterpret_cast<int *>(data);
            if(*ctrl_status == static_cast<int>(ActorCmd::CmdStop)) {
                this->actore_status_.store(static_cast<int>(ActorStatus::StatusStopped));
            }
            else if (*ctrl_status == static_cast<int>(ActorCmd::CmdPause))
            {
                this->actore_status_.store(static_cast<int>(ActorStatus::StatusPaused));
            }
            else if (*ctrl_status == ActorCmd::CmdRun)
            {
                this->actore_status_.store(static_cast<int>(ActorStatus::StatusRunning));
            }
            });
        if (!ret) {
            _mm_pause();
        }
    }
}

void AbstractActor::SetStatus(ActorStatus status) {
    actore_status_.store(static_cast<int>(status));
}

int AbstractActor::GetStatus() {
    return actore_status_.load();
}

void AbstractActor::SetStopFn(StopFn stop_fn) {
    stop_fn_ = stop_fn;
}

void AbstractActor::Run() {
    InitStatus();
    while(true) {
      if (actore_status_.load() >= static_cast<int>(ActorStatus::StatusStopped)) {
        break;
      }
      if (actore_status_.load() == static_cast<int>(ActorStatus::StatusRunning)) {
        target_fn_(this);
        LAVA_LOG(LOG_MP, "Actor: ActorStatus:%d ActorCommand: %d\n", GetStatus(), GetCmd());
        break;
      } else {
        // pause status
        _mm_pause();
      }
    }
    if (handle_cmd_thread_->joinable()) {
        handle_cmd_thread_->join();
    }
    if (stop_fn_ != NULL && actore_status_.load() != static_cast<int>(ActorStatus::StatusTerminated)) {
        stop_fn_();
    }
    LAVA_LOG(LOG_ACTOR, "child exist, pid:%d\n", this->pid_);
}

void AbstractActor::InitStatus() {
    actore_status_.store(static_cast<int>(ActorStatus::StatusRunning));
    handle_cmd_thread_ = std::make_shared<std::thread>(&AbstractActor::HandleCmd, this);
}

} // namespace message_infrastructure