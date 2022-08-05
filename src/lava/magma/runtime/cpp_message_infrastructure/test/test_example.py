from PyWrapper import MultiProcessing
import time


def print_hello():
    print("hello")
def main():
    mp = MultiProcessing()
    for i in range(5):
        mp.build_actor(print_hello)

    mp.check_actor()

main()
print("sleep 5")
time.sleep(5)
