import cv2
import numpy as np
import open3d as o3d
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.core.display_functions import display
import time
from scipy.spatial.transform import Rotation
import os
import csv

from utils import display_point_clouds

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)
########## Parameters ##########
xDebugLoading = False
xDebugSurfaceReconstruction = False
xDebugPlaneSegmenting = False
xDebugClustering = False
xDebugProcessing = False

sModelPath = os.path.join("Input", "T-stuk-filled.stl")
sScenePath = os.path.join("PointCloudImages/PointClouds_2024-11-27_19-41-14/2024-11-27_19-41-31/PointCloud_2024-11-27_19-41-31.ply")

## Surface reconstruction
iVoxelSize = 2
iPoissonDepth = 9
iLaplaceSmoothingIter = 500

########## 1. Load model and scene ##########
## Loading model as mesh
mshModel = o3d.io.read_triangle_mesh(sModelPath)

## Sample mesh to create point cloud using poisson disk sampling
pcdModel = mshModel.sample_points_poisson_disk(number_of_points=3000)
# pcdModel.scale(0.1, [0, 0, 0])

## Loading scene
pcdSceneRaw = o3d.io.read_point_cloud(sScenePath)

if xDebugLoading:
    display_point_clouds([pcdSceneRaw], "Input scene", False)
    display_point_clouds([pcdModel], "Input model", True)


########## 2. Preprocessing of the scene ##########
tProcessingStart = time.time()

## Voxel down scene
pcdSceneDown = pcdSceneRaw.voxel_down_sample(voxel_size=iVoxelSize)

if xDebugProcessing:
    display_point_clouds([pcdSceneDown], "Input scene - voxel down", False)


## Remove outliers
pcdSceneFiltered, _ = pcdSceneDown.remove_statistical_outlier(nb_neighbors=30, std_ratio=0.01)

if xDebugProcessing:
    display_point_clouds([pcdSceneFiltered], "Input scene - filtered", False)

oFittedPlane, arrInliersIndex = pcdSceneFiltered.segment_plane(distance_threshold=30,
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
arrVerticesToRemove = arrDensities < np.quantile(arrDensities, 0.2)
mshSurfRec.remove_vertices_by_mask(arrVerticesToRemove)

if xDebugSurfaceReconstruction:
    display_point_clouds([mshSurfRec], "Mesh - removed low density", False)

## Smoothing mesh with laplacian filter
mshSurfRecSmooth = mshSurfRec.filter_smooth_taubin(number_of_iterations=iLaplaceSmoothingIter)
mshSurfRecSmooth.compute_vertex_normals()
if xDebugSurfaceReconstruction:
    display_point_clouds([mshSurfRecSmooth], "Mesh - removed low density + taubin smooth", False)

pcdSceneROI = mshSurfRecSmooth.sample_points_poisson_disk(number_of_points=2000)

## Visualize final ROI pointcloud
if xDebugSurfaceReconstruction:
    display_point_clouds([pcdSceneROI], "Input scene - ROI - Final pointcloud", True)

tProcessingEnd = time.time()

print(f"Processing took {tProcessingEnd - tProcessingStart:.4f} seconds to execute.")

########## 3. Global registration using Open3D ##########

### Initial transformation
arrROIPoints = np.asarray(pcdSceneROI.points)
meanROI= np.mean(arrROIPoints, axis=0)

arrModelPoints = np.asarray(pcdModel.points)
meanModel = np.mean(arrModelPoints, axis=0)

arrInitTranslation = meanROI - meanModel

transformation_init = np.eye(4)
transformation_init[:3, 3] = arrInitTranslation

print(transformation_init)

pcdModelInitTransformed = pcdModel.transform(transformation_init)

#display_point_clouds([pcdModelInitTransformed, pcdSceneROI], "Initial transformation", False)

## Downsample both pointclouds to match the density
iVoxelMatching = 4

pcdModelInitTransformed = pcdModelInitTransformed.voxel_down_sample(iVoxelMatching)
pcdSceneROI = pcdSceneROI.voxel_down_sample(iVoxelMatching)

display_point_clouds([pcdModelInitTransformed, pcdSceneROI], "Initial transformation - voxel down", False)

iRadiusFeature = iVoxelMatching * 5
oModelFPFH = o3d.pipelines.registration.compute_fpfh_feature(pcdModelInitTransformed,
                                                             o3d.geometry.KDTreeSearchParamHybrid(radius=iRadiusFeature, max_nn=500))

oSceneFPFH = o3d.pipelines.registration.compute_fpfh_feature(pcdSceneROI,
                                                             o3d.geometry.KDTreeSearchParamHybrid(radius=iRadiusFeature, max_nn=500))

iDistanceThreshold = iVoxelMatching * 2
print(":: RANSAC registration on downsampled point clouds.")
print("   we use a liberal distance threshold %.3f." % iDistanceThreshold)
tMatchingStart = time.time()

result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    source=pcdModelInitTransformed,
    target=pcdSceneROI,
    source_feature=oModelFPFH,
    target_feature=oSceneFPFH,
    mutual_filter=False,
    max_correspondence_distance=iDistanceThreshold,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
    ransac_n=3,
    checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                iDistanceThreshold),
    ],
    criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
)

tMatchingEnd = time.time()

print(f"Matching took {tMatchingEnd - tMatchingStart:.4f} seconds to execute.")

print(result)
print(result.transformation)

pcdModelTransformed = pcdModelInitTransformed.transform(result.transformation)

display_point_clouds([pcdModelTransformed, pcdSceneROI], "Result", False)

########## 4. Local registration using Open3D ##########
distance = [0.001, 0.005]

for iDistance in distance:
    print("Apply point-to-point ICP")
    resultICP = o3d.pipelines.registration.registration_icp(
        source=pcdModelTransformed,
        target=pcdSceneROI,
        max_correspondence_distance=iDistance,
        init=result.transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
    )
    print(resultICP)
    print("Transformation is:")
    print(resultICP.transformation)


    display_point_clouds([pcdModelTransformed.transform(resultICP.transformation), pcdSceneROI], "Result", False)

