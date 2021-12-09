#include <stdio.h>
#include "run.h"

void run(runState* rs){
    printf("run called with phase: %d\n",rs->phase);
}