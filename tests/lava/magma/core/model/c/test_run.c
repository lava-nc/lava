/* 
 * Copyright (C) 2021 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 * See: https://spdx.org/licenses/
 */

#include <stdio.h>
#include "run.h"

void run(runState* rs){
    printf("run called with phase: %d\n",rs->phase);
}