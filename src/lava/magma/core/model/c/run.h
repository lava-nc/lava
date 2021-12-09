/* 
 * Copyright (C) 2021 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 * See: https://spdx.org/licenses/
 */

#ifndef _RUN_H
#define _RUN_H

#include <stddef.h>
#include <stdint.h>

typedef struct runState {
    uint8_t phase;
} runState;

void run(runState* state);

#endif