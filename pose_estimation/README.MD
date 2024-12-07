# pose_estimation package
Under construction ...
## ExtrinsicTransformationAruco
Estimate the extrinsic transformation matrix (affine 3D transformation) 
using ArUco markers detected in 3D point cloud data.

#### Features:
- **Initialization**: Processes the input point cloud (`pcdScene`), reshapes it into a 2D image-like array, and prepares color data for visualization.
- **ArUco Detection**: Detects markers using OpenCV's `aruco` module and extracts corresponding camera coordinates for specific marker IDs.
- **Camera-to-World Mapping**: Maps detected camera coordinates to their corresponding world coordinates, incorporating depth information using the normal vector of a fitted plane from the point cloud.
- **Affine Transformation Matrix Estimation**: Computes the 4x4 transformation matrix using `cv2.estimateAffine3D` with camera and world coordinate correspondences.
- **3D Plane Normal Calculation**: Fits a plane to the point cloud using RANSAC and calculates its normal vector for depth alignment.

## PointCloudManager
Manages loading, preprocessing, clustering, and reconstructing point clouds for 3D scene analysis and object reconstruction.

#### Features:
- **Model Loading**: Loads a 3D model mesh and samples it into a point cloud.
- **Scene Loading and Preprocessing**: Loads a scene point cloud, applies voxel downsampling, removes outliers, and selects regions of interest (ROI) for further analysis.
- **Background Removal**: Removes the dominant plane from the scene using RANSAC, leaving only the objects.
- **Clustering**: Uses DBSCAN clustering to identify and isolate individual objects in the scene based on point density.
- **Surface Reconstruction**: Performs surface reconstruction on clustered objects using Poisson surface reconstruction and Taubin smoothing for enhanced surface quality.
- **Visualization**: Provides methods to display the scene, model, clusters, reconstructed objects, and scene with background removed.

## PoseEstimatorFPFH
Estimates the pose (transformation matrix) between a model and a scene point cloud using Fast Point Feature Histograms (FPFH) for feature matching and RANSAC-based registration, followed by ICP refinement.

#### Features:
- **Initial Transformation**: Aligns the model and scene point clouds using a mean-to-mean transformation to approximate the initial pose.
- **Voxel Downsampling**: Downsamples both model and scene point clouds to reduce the number of points and ensure comparable density for feature calculation.
- **FPFH Feature Calculation**: Computes FPFH features for both the model and scene point clouds to capture the local shape information around each point.
- **Feature Matching with RANSAC**: Performs RANSAC-based feature matching to find the initial transformation between the model and scene based on FPFH features.
- **ICP Refinement**: Optionally refines the transformation using Iterative Closest Point (ICP) for improved alignment based on point-to-point distances.
- **Transformation Matrix**: Outputs the final 4x4 transformation matrix that aligns the model to the scene.
