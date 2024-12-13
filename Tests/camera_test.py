from luxonis_camera import Camera
from threading import Thread, Event
import time

stop_event = Event()

def run_camera():
    while not stop_event.is_set():
        oCamera.runCamera()


oCamera = Camera(5)

t1 = Thread(target=oCamera.runCamera)
t1.start()

## Main thread
try:
    # Main thread logic, or simply wait for termination
    while True:
        print("Main thread is running")
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopping threads...")
    stop_event.set()

t1.join()


