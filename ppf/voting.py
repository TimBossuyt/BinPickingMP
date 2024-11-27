import numpy as np
import random
from viz import viz
import sys
import time
import matplotlib.pyplot as plt


def display_matrix_graphically(array, cmap="viridis", title="Matrix"):
    """
    Display a 2D NumPy array as a graphical matrix.

    Parameters:
        array (numpy.ndarray): The 2D array to display.
        cmap (str): Colormap for the visualization (default: "viridis").
        title (str): Title of the matrix display.
    """
    if not isinstance(array, np.ndarray) or array.ndim != 2:
        raise ValueError("Input must be a 2D NumPy array.")

    plt.figure(figsize=(6, 6))
    plt.imshow(array, cmap=cmap, aspect="equal")
    plt.colorbar(label="Value")
    plt.title(title)
    plt.xlabel("Columns")
    plt.ylabel("Rows")

    # Annotate each cell with its value
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            plt.text(j, i, f"{array[i, j]:.2f}", ha="center", va="center", color="white")

    plt.show()

class voting:
    def __init__(self, model_ppf, scene_ppf, sample_number):
        sys.stdout.flush()

        self.model_ppf = model_ppf
        self.scene_ppf = scene_ppf

        ## Only select a few points from the scene .size gives number of points in the pointcloud
        self.n_samples = sample_number
        self.samples = random.choices(range(self.scene_ppf.size), k=self.n_samples )

        ## Create 3D array of zeros:
        ## axis 1 = model reference points
        ## axis 2 = discrete rotation angles (alpha?)
        ## axis 3 = each sample gets its own accumulator
        self.accumulator = np.zeros([self.model_ppf.size, self.model_ppf.angle_steps, sample_number])

        ## Call pose clustering method
        print("Started matching")
        self.pose_clustering()
        print("Matching done")

        # print(self.accumulator.shape)
        # print(self.accumulator)
        self.R, self.t = self.get_pose(3)

        self.viz = viz(self.model_ppf.model, self.scene_ppf.model, self.R, self.t)


    def pose_clustering(self):
        tStart = time.time()
        for sample_id, index in enumerate(self.samples):
            sys.stdout.write(f"Clustering sample {sample_id}/{self.n_samples} with reference scene point {index}\n")
            self.vote_from_reference(index, sample_id)
            sys.stdout.flush()
        tEnd = time.time()
        print(f"Clustering took {tEnd - tStart:.4f} seconds")

    def get_pose(self, sample_id):
        model_idx_r, alpha = np.unravel_index(self.accumulator[:, :, sample_id].argmax(), self.accumulator[:, :, sample_id].shape)

        sample_accumulator = self.accumulator[:, :, sample_id]
        print(sample_accumulator)
        #display_matrix_graphically(sample_accumulator)

        print(np.max(sample_accumulator))

        translation = self.model_ppf.points[model_idx_r]

        R2g = self.model_ppf.rotate_2g(model_idx_r)
        R_x = self.model_ppf.rotate_alpha(alpha)

        return R_x@R2g, translation

    def vote_from_reference(self, scene_idx_r, sample_id):
        for scene_idx_i in range(self.scene_ppf.size):
            if scene_idx_i != scene_idx_r:
                F_s = self.scene_ppf.get_ppf(scene_idx_r, scene_idx_i)

                model_pairs = self.model_ppf.hashtable.get(F_s, None)
                if model_pairs:
                    for (model_idx_r, model_idx_i, alpha_model) in model_pairs:
                        ## alpha = angle between model and scene --> used to calculate transformation
                        alpha = self.scene_ppf.get_alpha(scene_idx_r, scene_idx_i) - alpha_model

                        ## Use model_idx_r and alpha for voting
                        self.accumulator[model_idx_r, self.alpha_index(alpha), sample_id] += 1

    def alpha_index(self, alpha):
        return np.floor((alpha / np.pi) * self.model_ppf.angle_steps).astype(int)
