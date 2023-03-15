# C pingpong

## Run instructions
One needs to run the p.py first, start the ./cprocess binary once seeing the prompt and hit the enter as indicated.

Two args can be given as the socket file names.

```bash
# p.py
python3 p.py c2py py2c

# cprocess, in another terminal window
./cprocess c2py py2c
```

## Notes on current TempChennel:
TempChannel uses socket file.

The Recv port will bind the socket file in initialization, listen in start() and accept in recv(). After established a connection, the port closes it immediately after reading from the socket.

The Send port will connect to the recv port in initialization and write to the socket in send().

Therefore,
1. The send port can only be initialized after corresponding Recv port called start()
2. In each round, the send port is used one-off. One needs to create a new TempChannel() and get the send port from it each time. (The send port will be initialized when accessing the .dst_port property at the first time)