from luxonis_camera import Camera
from threading import Thread, Event
import time
import open3d as o3d

stop_event = Event()

def run_camera():
    while not stop_event.is_set():
        oCamera.run()

def visualization_loop(vis):
    while not stop_event.is_set():
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.03)  # Render at ~30 FPS


oCamera = Camera(5)

t1 = Thread(target=oCamera.run)
t1.start()

## Main thread
try:
    # Main thread logic, or simply wait for termination
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    first = True

    # Start visualization loop in its own thread
    vis_thread = Thread(target=visualization_loop, args=(vis,))
    vis_thread.start()

    ## Initial delay to wait until camera is started
    time.sleep(10)

    while True:
        ## Fetch pointcloud data
        oPcd = oCamera.getColoredPointCloud()

        if first:
            vis.add_geometry(oPcd)
            first = False
        else:
            vis.update_geometry(oPcd)

except KeyboardInterrupt:
    print("Stopping threads...")
    stop_event.set()

t1.join()
vis_thread.join()

