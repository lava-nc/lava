// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef CORE_ABSTRACT_ACTOR_H_
#define CORE_ABSTRACT_ACTOR_H_

#include <message_infrastructure/csrc/channel/shmem/shm.h>
#include <message_infrastructure/csrc/core/utils.h>
#include <functional>
#include <string>
#include <memory>
#include <utility>
#include <thread>  // NOLINT

namespace message_infrastructure {

enum class ActorType {
  RuntimeActor = 0,
  RuntimeServiceActor = 1,
  ProcessModelActor = 2
};

enum class ActorStatus {
  StatusError = -1,
  StatusRunning = 0,
  StatusPaused = 1,
  StatusStopped = 2,
  StatusTerminated = 3,
};

enum class ActorCmd {
  CmdRun = 0,
  CmdStop = -1,
  CmdPause = -2
};

struct ActorCtrlStatus {
  ActorCmd cmd;
  ActorStatus status;
};

class AbstractActor {
 public:
  using ActorPtr = AbstractActor *;
  using TargetFn = std::function<void(ActorPtr)>;
  using StopFn = std::function<void(void)>;

  explicit AbstractActor(TargetFn target_fn);
  virtual ~AbstractActor() = default;
  virtual int ForceStop() = 0;
  virtual int Wait() = 0;
  virtual ProcessType Create() = 0;
  void Control(const ActorCmd cmd);
  ActorStatus GetStatus();
  bool SetStatus(ActorStatus status);
  void SetStopFn(StopFn stop_fn);
  int GetPid() {
    return pid_;
  }

 protected:
  void Run();
  int pid_ = -1;

 private:
  SharedMemoryPtr ctl_shm_;
  std::atomic<int> actor_status_;
  std::thread handle_cmd_thread_;
  TargetFn target_fn_ = nullptr;
  StopFn stop_fn_ = nullptr;
  void InitStatus();
  void HandleCmd();
};

using SharedActorPtr = std::shared_ptr<AbstractActor>;

}  // namespace message_infrastructure

#endif  // CORE_ABSTRACT_ACTOR_H_
