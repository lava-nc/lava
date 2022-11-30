// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <message_infrastructure/csrc/core/abstract_actor.h>
#include <message_infrastructure/csrc/core/utils.h>
#include <message_infrastructure/csrc/core/message_infrastructure_logging.h>

namespace message_infrastructure {

AbstractActor::AbstractActor(AbstractActor::TargetFn target_fn)
  : target_fn_(target_fn) {
  ctl_shm_ = GetSharedMemManagerSingleton()
    .AllocChannelSharedMemory<SharedMemory>(sizeof(int));
  ctl_shm_->Start();
}

void AbstractActor::Control(const ActorCmd cmd) {
  ctl_shm_->Store([cmd](void* data){
  auto ctrl_cmd = reinterpret_cast<ActorCmd *>(data);
  *ctrl_cmd = cmd;
  LAVA_DEBUG(LOG_MP, "Cmd Get: %d\n", static_cast<int>(cmd));
  });
}

void AbstractActor::HandleCmd() {
  while (actor_status_.load() < static_cast<int>(ActorStatus::StatusStopped)) {
    auto ret = ctl_shm_->Load([this](void *data){
      auto ctrl_status = reinterpret_cast<int *>(data);
      if (*ctrl_status == static_cast<int>(ActorCmd::CmdStop)) {
        this->actor_status_
          .store(static_cast<int>(ActorStatus::StatusStopped));
        LAVA_DEBUG(LOG_MP, "Stop Recieved\n");
      } else if (*ctrl_status == static_cast<int>(ActorCmd::CmdPause)) {
        this->actor_status_
          .store(static_cast<int>(ActorStatus::StatusPaused));
      } else if (*ctrl_status == static_cast<int>(ActorCmd::CmdRun)) {
        this->actor_status_
          .store(static_cast<int>(ActorStatus::StatusRunning));
      }
    });
    if (!ret) {
      helper::Sleep();
    }
  }
}

bool AbstractActor::SetStatus(ActorStatus status) {
    LAVA_DEBUG(LOG_MP, "Set Status: %d\n", static_cast<int>(status));
    auto const curr_status = actor_status_.load();
    if (curr_status >= static_cast<int>(ActorStatus::StatusStopped)
      && static_cast<int>(status) < curr_status) {
      return false;
    }
    actor_status_.store(static_cast<int>(status));
    return true;
}

ActorStatus AbstractActor::GetStatus() {
  return static_cast<ActorStatus>(actor_status_.load());
}

void AbstractActor::SetStopFn(StopFn stop_fn) {
  stop_fn_ = stop_fn;
}

void AbstractActor::Run() {
  InitStatus();
  while (true) {
    if (actor_status_.load() >= static_cast<int>(ActorStatus::StatusStopped)) {
      break;
    }
    if (actor_status_.load() == static_cast<int>(ActorStatus::StatusRunning)) {
      target_fn_(this);
      LAVA_LOG(LOG_MP, "Actor:ActorStatus:%d\n", static_cast<int>(GetStatus()));
    } else {
      // pause status
      helper::Sleep();
    }
  }
  if (handle_cmd_thread_.joinable()) {
    handle_cmd_thread_.join();
  }
  if (stop_fn_ != nullptr &&
    actor_status_.load() != static_cast<int>(ActorStatus::StatusTerminated)) {
    stop_fn_();
  }
  LAVA_LOG(LOG_ACTOR, "child exist, pid:%d\n", pid_);
}

void AbstractActor::InitStatus() {
  actor_status_.store(static_cast<int>(ActorStatus::StatusRunning));
  handle_cmd_thread_ = std::thread(&AbstractActor::HandleCmd, this);
}

}  // namespace message_infrastructure
