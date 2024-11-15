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

## Parameter gridsearch
print("Starting gridsearch")
tGridSearchStart = time.time()
results = {}

arrSamplingSteps = [1, 0.8, 0.6, 0.4, 0.2, 0.1]
arrDiscretizationSteps = [0.8, 0.4, 0.2, 0.1, 0.05, 0.01]

for iRelativeSceneSampleStep in arrSamplingSteps:
    for iDiscretizationStep in arrDiscretizationSteps:
        ## Matching and calculating time
        tMatchingStart = time.time()
        estimator.match(iRelativeSceneSampleStep, iRelativeSceneDistance)
        tMatchingEnd = time.time()

        tMatchingDuration = tMatchingEnd - tMatchingStart

        arrPotentialPoses = estimator.getNPoses(5)

        first_pose_vote = None
        iVotesFirst = arrPotentialPoses[0].numVotes
        iVotesSecond = arrPotentialPoses[1].numVotes

        for idx, pose in enumerate(arrPotentialPoses):
            pose.printPose()

            pcdModelTransformed = o3d.geometry.PointCloud()
            pcdModelTransformed.points = pcdModel.points
            pcdModelTransformed.transform(np.asarray(pose.pose))
            pcdModelTransformed.paint_uniform_color([0, 1, 0])
            display_point_clouds([pcdModelTransformed, pcdSceneFiltered], "Result", False)

            if idx == 0:
                ## Ask user if first pose is correct
                first_pose_vote = input("Is the first pose correct? (y/n): ").strip().lower()
                break

        results[iRelativeSceneSampleStep, iDiscretizationStep] = {
            "first-pose-correct": first_pose_vote,
            "first-pose-votes": iVotesFirst,
            "second-pose-votes": iVotesSecond,
            "matching-duration": tMatchingDuration
        }

tGridSearchEnd = time.time()
tGridsearchDuration = tGridSearchEnd - tGridSearchStart
print(f"Gridsearch took {tGridsearchDuration:.4f} seconds to execute.")
print(results)

sOutputFilename = "gridsearch_results.csv"
csv_headers = ["Sampling Step", "Discretization Step", "First Pose correct?", "Votes no. 1","Votes no. 2", "Matching Duration"]

# Open the file and write the results
with open(sOutputFilename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(csv_headers)  # Write the header row

    for (sampling_step, discretization_step), metrics in results.items():
        row = [
            sampling_step,
            discretization_step,
            metrics["first-pose-correct"],
            metrics["first-pose-votes"],
            metrics["second-pose-votes"],
            metrics["matching-duration"],
        ]
        writer.writerow(row)

print(f"Results saved to {sOutputFilename}")
