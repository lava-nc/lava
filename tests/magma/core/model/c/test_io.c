#include <unistd.h>
#include <stdio.h>
#include "run.h"
#include "ports.h"

void run(runState* rs){
    printf("run called with phase: %d\n",rs->phase);
    Port *in = get_port("in_port");
    Port *out = get_port("out_port");
    printf("got ports\n");
    int *buf;
    while(!peek(in))sleep(1);
    printf("ready to recieve:%p\n",&buf);
    recv(in,&buf);
    printf("recieved: %p\n",buf);
    while(!probe(out))sleep(1);
    printf("ready\n");
    send(out,*buf,1);
    printf("sent\n");
}