// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <core/abstract_channel.h>
#include <core/channel_factory.h>
#include <core/abstract_port.h>
#include <core/utils.h>
#include <channel/socket/socket_port.h>

using namespace message_infrastructure; // NOLINT

int main(int argc, char *argv[])
{

    char *c2py = (argc >= 2) ? argv[1] : (char *)"./c2py";
    char *py2c = (argc >= 3) ? argv[2] : (char *)"./py2c";

    std::cout << "socket files: " << c2py << " " << py2c << "\n";

    ChannelFactory &channel_factory = GetChannelFactory();

    AbstractChannelPtr ch = channel_factory.GetTempChannel(py2c);
    AbstractRecvPortPtr rc = ch->GetRecvPort();

    // order matters
    rc->Start();

    for (uint _ = 0; _ < 10; ++_)
    {
        std::cout << "receiving\n";
        MetaDataPtr recvd = rc->Recv();
        std::cout << "received from py, total size: "
                  << recvd->total_size
                  << "\n";

        AbstractChannelPtr ch2 = channel_factory.GetTempChannel(c2py);
        AbstractSendPortPtr sd = ch2->GetSendPort();
        sd->Start();
        sd->Send(recvd);
        sd->Join();
    }

    rc->Join();

    return 0;
}
