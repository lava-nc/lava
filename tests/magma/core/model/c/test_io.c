#include <unistd.h>
#include <stdio.h>
#include "run.h"
#include "ports.h"

void run(runState* rs){
    printf("run called with phase: %d\n",rs->phase);
    Port *port = get_port("port");
    printf("got port %p\n",port);
    int *buf;
    while(!peek(port))sleep(1);
    printf("ready to recieve at %p\n",&buf);
    recv(port,&buf);
    printf("recieved: %p\n",buf);
    while(!probe(port))sleep(1);
    printf("ready\n");
    send(port,buf,1);
    printf("sent\n");
}