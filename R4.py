import numpy as np

class TransformationMatrix:
    def __init__(self, R3_filename):
        self.R3_filename = R3_filename
        self.R3_matrix = self.load_matrix(self.R3_filename)
        self.pose_matrix = None  # To be set with `update_pose_matrix`
        self.R4 = None

    # Function to load a 4x4 matrix from a .txt file
    def load_matrix(self, filename):
        with open(filename, 'r') as file:
            matrix = [list(map(float, line.strip().split())) for line in file]
            return np.array(matrix)

    # Function to set pose_matrix with the pose obtained from the estimation
    def update_pose_matrix(self, pose):
        self.pose_matrix = pose
        self.R4 = self.compute_base_to_object()

    # Function to compute the base-to-object transformation
    def compute_base_to_object(self):
        if self.pose_matrix is not None:
            return np.dot(self.R3_matrix, self.pose_matrix)
        else:
            raise ValueError("Pose matrix has not been set.")

    # Function to get the resulting base-to-object transformation matrix
    def get_base_to_object(self):
        if self.R4 is not None:
            return self.R4
        else:
            raise ValueError("Base-to-object transformation matrix has not been computed yet.")


# to be checked.....