# Camera Module
The `Camera` class interfaces with an OAK-D device to capture RGB images and generate point clouds.

## Camera
The `Camera` class manages the DepthAI pipeline, handles image and point cloud requests, and runs a separate thread to process data asynchronously.

#### Features:
- **Pipeline Configuration**: Initializes a DepthAI pipeline with a color camera and stereo depth cameras.
- **Multi-threaded Operation**: Runs a separate thread to process data while responding to requests using a queue-based communication system.
- **RGB Image Capture**: Provides access to the latest RGB preview frame and high-resolution image frames upon request.
- **Colored Point Cloud Generation**: Generates a 3D point cloud with corresponding RGB colors using synchronized depth and color information.
- **Queue-based Request Handling**: Uses request and response queues to manage inter-thread communication efficiently.

## Methods
### `getColoredPointCloud()`
Retrieves a 3D colored point cloud from the device.
- **Sends a request** to the processing thread.
- **Receives processed point cloud data** (3D points and color values).
- **Creates an Open3D point cloud** with the received data.

### `getCvVideoPreview()`
Returns the latest RGB preview frame.
- Used for real-time preview.
- Continuously updated in the background.

### `getCvImageFrame()`
Retrieves a high-resolution RGB image from the device.
- **Sends a request** for an image frame.
- **Receives the captured frame** from the processing thread.

### `Stop()`
Stops the camera processing thread by setting a termination event.

### `Run()`
Starts the camera processing in a separate thread.
- Calls the internal `run_()` method.

### `run_()`
Processes incoming requests and captures data from the OAK-D device.
- Retrieves synchronized RGB and depth data.
- Handles queued requests for images and point clouds.

### `_configurePipeline()`
Configures the DepthAI pipeline:
- **Color Camera**: Captures 1080p images.
- **Stereo Cameras**: Capture depth information with 800p resolution.
- **Stereo Depth Node**: Computes depth maps using high-accuracy settings.
- **Point Cloud Node**: Generates 3D point clouds from depth data.
- **Sync Node**: Synchronizes RGB images with depth data.
- **Output Streams**: Sends RGB and point cloud data to the host.


