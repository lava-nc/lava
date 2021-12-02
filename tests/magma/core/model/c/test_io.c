#include <unistd.h>
#include <stdio.h>
#include "run.h"
#include "ports.h"

void run(runState* rs){
    Port *in = get_port("in_port");
    Port *out = get_port("out_port");
    int *buf;
    while(!peek(in))sleep(1);
    recv(in,&buf);
    while(!probe(out))sleep(1);
    send(out,*buf,1);
}