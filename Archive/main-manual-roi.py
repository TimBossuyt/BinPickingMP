import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import time
import os
from utils import display_point_clouds, extract_square_region, PoseEstimator

########## Parameters ##########
xDebugLoading = False
xDebugROI = False
xDebugPreProcessing = False
xDebugSurfaceReconstruction = False

sModelPath = os.path.join("Input", "astronaut.stl")
sScenePath = os.path.join("Input", "scene-astronaut.ply")

## For saving ply files
sProcessingFolder = "Processing"
sModelName = "model.ply"
sSceneName = "scene.ply"

## Image resolution: 1920x1080
iWidthImage = 1920
iHeightImage = 1080

## ROI Parameters (manual using matplotlib image)
iLocX = 930
iLocY = 490
iModelDiameter = 200


########## 1. Load model and scene ##########
## Loading model as mesh
mshModel = o3d.io.read_triangle_mesh(sModelPath)

## Sample mesh to create point cloud using poisson disk sampling
pcdModel = mshModel.sample_points_poisson_disk(number_of_points=5000)

## Loading scene
pcdSceneRaw = o3d.io.read_point_cloud(sScenePath)

if xDebugLoading:
    display_point_clouds([pcdSceneRaw], "Input scene", False)
    display_point_clouds([pcdModel], "Input model", True)

########## 2. Preprocessing of the scene ##########
tProcessingStart = time.time()

## Select ROI
arrColors = np.asarray(pcdSceneRaw.colors).reshape(iHeightImage, iWidthImage, 3) # (1920, 1080, 3), RGB
arrColorsBGR = arrColors[:, :, ::-1] # RGB to BGR

if xDebugROI: # Show matplotlib figure of image to determine center of ROI
    plt.figure()
    plt.imshow(arrColors)
    plt.show()

arrPcdScenePoints = np.asarray(pcdSceneRaw.points)
# Get points inside square defined by center and diameter
arrPointsROI = extract_square_region(arrPcdScenePoints.reshape(iHeightImage, iWidthImage, 3), (iLocX, iLocY), iModelDiameter*2)
arrPointsROIFlatten = arrPointsROI.reshape(-1, 3) ## Flatten to create new point cloud

pcdSceneROI = o3d.geometry.PointCloud()
pcdSceneROI.points = o3d.utility.Vector3dVector(arrPointsROIFlatten)
pcdSceneROI.paint_uniform_color([0, 0, 0])

if xDebugPreProcessing:
    display_point_clouds([pcdSceneROI], "Scene ROI", False)


## Remove outliers and downsample using Voxel
pcdSceneROI, _ = pcdSceneROI.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.5)

if xDebugPreProcessing:
    display_point_clouds([pcdSceneROI], "Scene ROI - Removed outliers", False)

pcdSceneROI = pcdSceneROI.voxel_down_sample(voxel_size=3)

if xDebugPreProcessing:
    display_point_clouds([pcdSceneROI], "Scene ROI - Voxel Down", False)

## Surface reconstruction, mesh and poisson disk sampling
#Estimate normals for the point cloud (required for Poisson reconstruction)
pcdSceneROI.estimate_normals() # Point cloud needs to have normals in order to point them towards the camera
pcdSceneROI.orient_normals_towards_camera_location([0, 0, 0])
pcdSceneROI.estimate_normals()

if xDebugSurfaceReconstruction:
    display_point_clouds([pcdSceneROI], "Scene ROI - Normal estimates", True)

# Create a mesh from the point cloud using Poisson reconstruction
mshSceneROI, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd=pcdSceneROI, depth=15)
mshSceneROI.orient_triangles()
mshSceneROI.compute_vertex_normals()

if xDebugSurfaceReconstruction:
    display_point_clouds([mshSceneROI], "Scene ROI - Mesh", False)

# Apply Laplacian smoothing on the mesh
mshSceneROI = mshSceneROI.filter_smooth_laplacian(number_of_iterations=100)

if xDebugSurfaceReconstruction:
    display_point_clouds([mshSceneROI], "Scene ROI - Mesh with smoothing", False)

pcdSceneROI = mshSceneROI.sample_points_poisson_disk(number_of_points=5000)

if xDebugSurfaceReconstruction:
    display_point_clouds([pcdSceneROI], "Scene ROI - Sampled", True)

## Saving the ply files to use with surface matching
o3d.io.write_point_cloud(os.path.join(sProcessingFolder, sModelName), pcdModel, write_ascii=True)
o3d.io.write_point_cloud(os.path.join(sProcessingFolder, sSceneName), pcdSceneROI, write_ascii=True)

tProcessingEnd = time.time()

print(f"Processing took {tProcessingEnd - tProcessingStart:.4f} seconds to execute.")

########## 3. Train model / calculate PPF ##########
estimator = PoseEstimator(0.05, 0.03)

estimator.loadModel(os.path.join(sProcessingFolder, sModelName), 1)
estimator.loadScene(os.path.join(sProcessingFolder, sSceneName), 1)

tTrainingStart = time.time()

estimator.trainModel()

tTrainingEnd = time.time()

print(f"Training took {tTrainingEnd - tTrainingStart:.4f} seconds to execute.")

## Saving model
# estimator.savePPF("PointCloudImages", "model")

########## 4. Match model to scene ##########
tMatchingStart = time.time()

estimator.match(0.020, 0.05)

tMatchingEnd = time.time()

print(f"Matching took {tMatchingEnd - tMatchingStart:.4f} seconds to execute.")

########## 5. Show results ##########
# estimator.showResults(5)

### Show resulting pose on original scene
arrPotentialPoses = estimator.getNPoses(5)

for pose in arrPotentialPoses:
    pose.printPose()

    pcdModelTransformed = o3d.geometry.PointCloud()
    pcdModelTransformed.points = pcdModel.points
    pcdModelTransformed.transform(np.asarray(pose.pose))
    pcdModelTransformed.paint_uniform_color([0, 1, 0])

    display_point_clouds([pcdModelTransformed, pcdSceneROI], "Result", False)