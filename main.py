import cv2
import numpy as np
import open3d as o3d
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.core.display_functions import display
from scipy.spatial.transform import Rotation
import time
import os
import csv

from utils import display_point_clouds, extract_square_region, PoseEstimator

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)
########## Parameters ##########
xDebugLoading = False
xDebugSurfaceReconstruction = False
xDebugPlaneSegmenting = False
xDebugClustering = False
xDebugProcessing = False

sModelPath = os.path.join("Input", "astronaut-mesh-ok.stl")
sScenePath = os.path.join("PointCloudImages/PointClouds_2024-11-12_15-40-48/2024-11-12_15-40-59/PointCloud_2024-11-12_15-40-59.ply")

## For saving ply files
sProcessingFolder = "Processing"
sModelName = "model.ply"
sSceneName = "scene.ply"

## Training
iRelativeSamplingStepModel = 0.05
iRelativeDistanceStepModel = 0.025
iNumOfAngles = 25

## Surface reconstruction
iVoxelSize = 2
iPoissonDepth = 9
iLaplaceSmoothingIter = 200

## Matching
iRelativeSceneSampleStep = 0.01
iRelativeSceneDistance = 0.01

########## 1. Load model and scene ##########
## Loading model as mesh
mshModel = o3d.io.read_triangle_mesh(sModelPath)

## Sample mesh to create point cloud using poisson disk sampling
pcdModel = mshModel.sample_points_poisson_disk(number_of_points=5000)
pcdModel.scale(0.1, [0, 0, 0])

## Loading scene
pcdSceneRaw = o3d.io.read_point_cloud(sScenePath)

if xDebugLoading:
    display_point_clouds([pcdSceneRaw], "Input scene", False)
    display_point_clouds([pcdModel], "Input model", True)


########## 2. Train model / calculate PPF ##########
estimator = PoseEstimator(iRelativeSamplingStepModel, iRelativeDistanceStepModel, iNumOfAngles)

estimator.loadModel(os.path.join(sProcessingFolder, sModelName), 1)
estimator.loadScene(os.path.join(sProcessingFolder, sSceneName), 1)

tTrainingStart = time.time()

estimator.trainModel()

tTrainingEnd = time.time()

print(f"Training took {tTrainingEnd - tTrainingStart:.4f} seconds to execute.")

## Saving the ply files to use with surface matching
o3d.io.write_point_cloud(os.path.join(sProcessingFolder, sModelName), pcdModel, write_ascii=True)

########## 3. Preprocessing of the scene ##########
tProcessingStart = time.time()

## Voxel down scene
pcdSceneDown = pcdSceneRaw.voxel_down_sample(voxel_size=iVoxelSize)

if xDebugProcessing:
    display_point_clouds([pcdSceneDown], "Input scene - voxel down", False)


## Remove outliers
pcdSceneFiltered, _ = pcdSceneDown.remove_statistical_outlier(nb_neighbors=30, std_ratio=0.01)

if xDebugProcessing:
    display_point_clouds([pcdSceneFiltered], "Input scene - filtered", False)

oFittedPlane, arrInliersIndex = pcdSceneFiltered.segment_plane(distance_threshold=20,
                                                               ransac_n=100,
                                                               num_iterations=1000,
                                                               probability=0.9999)

pcdSceneNoBackground = pcdSceneFiltered.select_by_index(arrInliersIndex, invert=True)

if xDebugPlaneSegmenting:
    pcdBackground = pcdSceneFiltered.select_by_index(arrInliersIndex) ## Get pointcloud of all points on plane
    pcdBackground = pcdBackground.paint_uniform_color([1.0, 0, 0]) ## Paint red
    display_point_clouds([pcdBackground, pcdSceneNoBackground], "Input scene - plane segmenting", False)

## DBSCAN Clustering
## Get label by index of XYZ-point
arrLabels = np.array(pcdSceneNoBackground.cluster_dbscan(eps=10, min_points=10, print_progress=True))
clusters = {}
print(f"Number of detected clusters: {len(set(arrLabels))}") ## set() gets all different labels

## Getting the largest cluster (only for 1 object detection)
iMostPoints = 0
arrLargestCluster = None

for label in set(arrLabels):
    ## Create dictionary with key = label and values = list of points with that label
    clusters[label] = np.asarray(pcdSceneNoBackground.points)[arrLabels == label]

    iNumberOfPoints = len(clusters[label]) ## Number of points in cluster
    if iNumberOfPoints > iMostPoints:
        arrLargestCluster = clusters[label] ## Assign largest cluster
        iMostPoints = iNumberOfPoints

if xDebugClustering:
    max_label = arrLabels.max() ## Get max value of label
    colors = plt.get_cmap("tab20")(arrLabels / (max_label if max_label > 0 else 1)) ## create colormap
    colors[arrLabels < 0] = 0  # Noise points set to black
    pcdSceneNoBackground.colors = o3d.utility.Vector3dVector(colors[:, :3]) ## assign different color to each cluster

    display_point_clouds([pcdSceneNoBackground], "Clustered Point Cloud", False)

## Select largest cluster as ROI
pcdSceneROI = o3d.geometry.PointCloud()
pcdSceneROI.points = o3d.utility.Vector3dVector(arrLargestCluster)

## Calculating normals
pcdSceneROI.estimate_normals() # Point cloud needs to have normals in order to point them towards the camera
pcdSceneROI.orient_normals_towards_camera_location([0, 0, 0])
pcdSceneROI.estimate_normals()

if xDebugSurfaceReconstruction:
    display_point_clouds([pcdSceneROI], "Input scene - ROI - Before smoothing", True)

## Smoothing
print('run Poisson surface reconstruction')
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    mshSurfRec, arrDensities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd=pcdSceneROI, depth=iPoissonDepth)

arrDensities = np.asarray(arrDensities)

if xDebugSurfaceReconstruction:
    ## Visualize poisson sampling densities
    print('visualize densities')
    densities = arrDensities
    density_colors = plt.get_cmap('plasma')(
        (densities - densities.min()) / (densities.max() - densities.min()))
    density_colors = density_colors[:, :3]
    density_mesh = o3d.geometry.TriangleMesh()
    density_mesh.vertices = mshSurfRec.vertices
    density_mesh.triangles = mshSurfRec.triangles
    density_mesh.triangle_normals = mshSurfRec.triangle_normals
    density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)
    display_point_clouds([density_mesh], "Density mesh visualization", False)

## Removing points with low density
arrVerticesToRemove = arrDensities < np.quantile(arrDensities, 0.1)
mshSurfRec.remove_vertices_by_mask(arrVerticesToRemove)

if xDebugSurfaceReconstruction:
    display_point_clouds([mshSurfRec], "Mesh - removed low density", False)

## Smoothing mesh with laplacian filter
mshSurfRecSmooth = mshSurfRec.filter_smooth_laplacian(number_of_iterations=iLaplaceSmoothingIter)
mshSurfRecSmooth.compute_vertex_normals()
if xDebugSurfaceReconstruction:
    display_point_clouds([mshSurfRecSmooth], "Mesh - removed low density + laplacian smooth", False)

pcdSceneROI = mshSurfRecSmooth.sample_points_poisson_disk(number_of_points=3000)

# pcdSceneROI = pcdSceneROI.voxel_down_sample(voxel_size=4)

# pcdSceneROI.estimate_normals()

## Visualize final ROI pointcloud
if xDebugSurfaceReconstruction:
    display_point_clouds([pcdSceneROI], "Input scene - ROI - Final pointcloud", True)

## Saving the ply files to use with surface matching
o3d.io.write_point_cloud(os.path.join(sProcessingFolder, sSceneName), pcdSceneROI, write_ascii=True)

tProcessingEnd = time.time()

print(f"Processing took {tProcessingEnd - tProcessingStart:.4f} seconds to execute.")


########## 4. Match model to scene ##########
tMatchingStart = time.time()

estimator.match(iRelativeSceneSampleStep, iRelativeSceneDistance)

tMatchingEnd = time.time()

print(f"Matching took {tMatchingEnd - tMatchingStart:.4f} seconds to execute.")

########## 5. Show results ##########

### Show resulting pose on original scene
arrPotentialPoses = estimator.getNPoses(5)

for pose in arrPotentialPoses:
    pose.printPose()

    pcdModelTransformed = o3d.geometry.PointCloud()
    pcdModelTransformed.points = pcdModel.points
    pcdModelTransformed.transform(np.asarray(pose.pose))
    pcdModelTransformed.paint_uniform_color([0, 1, 0])

    display_point_clouds([pcdModelTransformed, pcdSceneFiltered], "Result", False)