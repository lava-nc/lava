#include "lava/magma.h"


struct Shared {
    char semaphore;
    void* data;
};

char data[100]
struct Shared state = {0,data};

void run(uint16 phase){
    switch(phase){
        case 0:
            if(probe(state))
                send(state)
            break;
        case 1:
            if(peek(state)){ 
                recv(state);
                inc();
            }
            break;
        default:
            break; 
    }

}

void inc(){
    for(int i=0;i<100;++i)data[i]+=1;
}