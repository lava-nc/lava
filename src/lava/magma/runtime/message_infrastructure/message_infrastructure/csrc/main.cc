#include "shmem_channel.h"
#include "shmem_port.h"
#include "channel_proxy.h"
#include "utils.h"

using namespace message_infrastructure;

int main()
{
  int data[3];
  for(int i = 0; i < 3; i++)
    data[i] = i;
  int *res;
  SharedMemManager smm;
  ChannelFactory& channel_factory = GetChannelFactory();
  std::shared_ptr<AbstractChannel> channel = channel_factory.GetChannel(SHMEMCHANNEL, smm, 2, 3 *4);
  SendPortProxyPtr send_port = channel->GetSendPort();
  RecvPortProxyPtr recv_port = channel->GetRecvPort();

  send_port->Start();
  recv_port->Start();

  send_port->Send(data);
  res = (int*)recv_port->Recv();
  for(int i = 0; i < 3; i++)
    printf("%i \n",res[i]);
  return 0;
}
