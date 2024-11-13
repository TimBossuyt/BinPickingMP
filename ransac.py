import cv2
import numpy as np
import open3d as o3d
import open3d.core as o3c
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.core.display_functions import display
from scipy.spatial.transform import Rotation
import time
import os
from utils import display_point_clouds, extract_square_region, PoseEstimator
import copy


print(o3d.__version__)
########## Parameters ##########
xDebugLoading = True

sScenePath = os.path.join("Input", "scene-astronaut-ransac.ply")

## Loading scene
pcdSceneRaw = o3d.io.read_point_cloud(sScenePath)

if xDebugLoading:
    display_point_clouds([pcdSceneRaw], "Input scene", False)

########## 2. Preprocessing of the scene ##########
pcdSceneDown = pcdSceneRaw.voxel_down_sample(voxel_size=2)

## Remove outliers first
pcdSceneFiltered, _ = pcdSceneDown.remove_statistical_outlier(nb_neighbors=30, std_ratio=0.01)

display_point_clouds([pcdSceneFiltered], "Input scene - filtered", False)

plane_model, inliers = pcdSceneFiltered.segment_plane(distance_threshold=20,
                                            ransac_n=100,
                                            num_iterations=1000,
                                            probability=0.9999)

[a, b, c, d] = np.asarray(plane_model)
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

pcdInliers = pcdSceneFiltered.select_by_index(inliers)
pcdInliers = pcdInliers.paint_uniform_color([1.0, 0, 0])
pcdOutliers = pcdSceneFiltered.select_by_index(inliers, invert=True)
display_point_clouds([pcdInliers, pcdOutliers], "Input scene - plane segmenting", False)

display_point_clouds([pcdOutliers], "Input scene - Outliers", False)

labels = np.array(pcdOutliers.cluster_dbscan(eps=10, min_points=10, print_progress=True))
clusters = {}
print(f"Number of detected clusters: {len(set(labels))}")

iMostPoints = 0
arrLargestCluster = None

for label in set(labels):
    clusters[label] = np.asarray(pcdOutliers.points)[labels == label]

    iNumberOfPoints = len(clusters[label])

    if iNumberOfPoints > iMostPoints:
        arrLargestCluster = clusters[label]
        iMostPoints = iNumberOfPoints

## Set color for the clusters
max_label = labels.max()
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0  # Noise points set to black
pcdOutliers.colors = o3d.utility.Vector3dVector(colors[:, :3])

o3d.visualization.draw_geometries([pcdOutliers], window_name="Clustered Point Cloud")

## Only show the largest cluster
pcdLargestCluster = o3d.geometry.PointCloud()
pcdLargestCluster.points = o3d.utility.Vector3dVector(arrLargestCluster)

o3d.visualization.draw_geometries([pcdLargestCluster], window_name="Clustered Point Cloud - Largest Cluster")

