// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#ifndef SHMEM_PORT_H_
#define SHMEM_PORT_H_

#include "abstract_port.h"

namespace message_infrastrature {

class ShmemSendPort : public AbstractSendPort {

};

class ShmemRecvPort : public AbstractRecvPort {

};

} // namespace message_infrastrature

#endif
