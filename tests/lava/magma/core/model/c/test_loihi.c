#include "ports.h"

int pre_guard(){
    return 0;
}

int run_pre_mgmt(){
    return 0;
}

int lrn_guard(){
    return 0;
}

int run_lrn(){
    return 0;
}

int post_guard(){
    return 0;
}

int run_post_mgmt(){
    return 0;
}

int host_guard(){
    return 0;
}

int run_host_mgmt(){
    return 0;
}

int run_spk(){
    Port* p_in = get_port("s_in");
    Port* p_out = get_port("a_out");
    void** data;
    recv(p_in,data);
    send(p_out,*data,1);
    flush(p_out);
    return 0;
}
