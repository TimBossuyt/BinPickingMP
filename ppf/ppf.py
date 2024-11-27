import numpy as np
import open3d as o3d
from udtHashTable import HashTable
import time
import sys

class ppf:
    def __init__(self, cloud_filename, voxel_size, distance_sampling_rate, angle_steps):
        self.input = o3d.io.read_point_cloud(cloud_filename)
        print(f"Loaded input of {np.asarray(self.input.points).shape[0]} points")

        # Estimate normals using a radius-based search
        self.norm_parm = o3d.geometry.KDTreeSearchParamKNN()
        self.input.estimate_normals()

        # Consistent normal orientation
        self.input.orient_normals_consistent_tangent_plane(k = 50)

        # Downsample the point cloud
        self.model = self.input.voxel_down_sample(voxel_size)
        self.points = np.asarray(self.model.points)
        self.normals = np.asarray(self.model.normals)

        # Initialize KD-tree for fast nearest neighbor search
        self.tree = o3d.geometry.KDTreeFlann(self.model)
        self.size = self.points.shape[0] ## Number of points in pointcloud
        print(f"Number of points after downsampling: {self.size}")

        ## Discretization parameters
        self.distance_sampling_rate = distance_sampling_rate
        self.model_diameter = np.max(np.max(self.points, axis=0) - np.min(self.points, axis=0))
        self.d_dist = np.round(self.distance_sampling_rate * self.model_diameter, 0)
        self.angle_steps = angle_steps
        self.d_angle = np.round(2*np.pi/self.angle_steps, 5)

        print(f"Model has a maximum diameter of {self.model_diameter}")

        # Initialize hash table for storing PPFs
        self.hashtable = HashTable()
        self.bTrained = False

    def vis(self):
        o3d.visualization.draw_geometries([self.model])

    def angle(self, vec1, vec2):
        """
        Computes the angle between two vectors using the tangent-based approach for numerical stability.
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        cross_product = np.cross(vec1, vec2)
        magnitude_cross = np.linalg.norm(cross_product)
        dot_product = np.dot(vec1, vec2)

        angle_rad = np.arctan2(magnitude_cross, dot_product)

        return angle_rad % np.pi

    def get_ppf(self, idx1, idx2):
        """
        Extracts the Point Pair Feature (PPF) for a pair of points in the model.
        """
        d = self.points[idx2] - self.points[idx1]

        dnorm = self.discrete_distance(np.linalg.norm(d))
        alpha = self.discrete_angle(self.angle(self.normals[idx1], d))
        beta = self.discrete_angle(self.angle(self.normals[idx2], d))
        gamma = self.discrete_angle(self.angle(self.normals[idx1], self.normals[idx2]))
        return dnorm, alpha, beta, gamma

    def discrete_distance(self, dist):
        """
        Discretizes the distance between two points
        """
        steps = dist // self.d_dist ## discrete steps to value

        return np.round(steps*self.d_dist)

    def discrete_angle(self, angle):
        """
        Discretizes the angle between two vectors
        """
        steps = angle // self.d_angle

        return np.round(steps*self.d_angle, 5)

    def make_hash_table(self):
        """
        Generates a hash table storing Point Pair Features (PPF) for all unique point pairs in the model.
        The hash table maps each PPF to a pair of point indices (idx1, idx2).
        """
        idx = 0

        tTrainingStart = time.time()

        total_pairs = self.size * self.size

        for idx1 in range(self.size):
            for idx2 in range(self.size):
                if idx1 != idx2:
                    ppf_keys = self.get_ppf(idx1, idx2)

                    if ppf_keys not in self.hashtable.keys():
                        ## If bin not already made --> list with index list
                        self.hashtable[ppf_keys] = [[idx1, idx2, self.get_alpha(idx1, idx2)]]
                    else:
                        ## If bin exists --> add indexes to bin
                        self.hashtable[ppf_keys].append([idx1, idx2, self.get_alpha(idx1, idx2)])

                idx += 1
                progress = (idx / total_pairs) * 100

                sys.stdout.write(f"\rProcessing pairs: {progress:.2f}% complete")
                sys.stdout.flush()

        tTrainingEnd = time.time()
        print(f"\nTraining took {tTrainingEnd - tTrainingStart:.4f} seconds to execute.")
        self.bTrained = True

    def rodrigues(self, angle, axis):
        """
        Computes the rotation matrix for rotating a vector by a given angle around a specified axis using Rodrigues' rotation formula.
        """
        if np.linalg.norm(axis) != 0:
            axis /= np.linalg.norm(axis)
        # skew symmetric form
        s_axis = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * s_axis + (1 - np.cos(angle)) * s_axis @ s_axis
        return R

    def rotate_2g(self, idx_r):
        """
        Computes the rotation matrix that rotates the normal of the reference point (idx_r) to align with the x-axis.
        """
        v1 = self.normals[idx_r]
        v2 = np.array([1, 0, 0])

        angle = np.arccos(min(v1.dot(v2), 1))
        axis = -np.cross(v1, v2)

        return self.rodrigues(angle, axis)

    def get_alpha(self, idx_r, idx_i):
        """
        Computes the relative angle between two points after aligning their normals with the x-axis.
        """
        # idx_r = index of reference point
        # idx_i = second point from pair

        ## 1. translate second point to reference point --> 2 points on same coordinate system
        pt_idx_i = self.points[idx_i] - self.points[idx_r]

        ## 2. Rotation to align normals, align idx_r normal with the x-axis
        R2g = self.rotate_2g(idx_r)
        ## 3. Rotate second point the same way
        pt_idx_i = R2g@(pt_idx_i.reshape(3, 1))
        ## 4. Project on y-z plane (ignoring x-component, already aligned with rotation)
        pt_idx_i[0]= 0

        ## 5. normalize
        pt_idx_i /= np.linalg.norm(pt_idx_i)

        ## 6. Return angle to y-axis
        return np.arccos(pt_idx_i.reshape(3).dot(np.array([0,1,0])))

    def save_hashtable(self, filename):
        if not self.bTrained:
            print("No PPF have been calculted yet")
            return

        with open(filename, 'w') as f:
            for key, values in self.hashtable.items():
                f.write(f"{key}\t")
                for pair in values:
                    f.write(f"{pair}\t")
                f.write('\n')

    def rotate_alpha(self, alpha):
        # rotation axis and matrix
        angle = alpha
        axis = np.array([1.0, 0.0, 0.0])
        return self.rodrigues(angle, axis)

