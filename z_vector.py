import numpy as np


def get_upward_and_position(transformation_matrix):
    # Extract the rotation part of the matrix
    rotation_matrix = transformation_matrix[:3, :3]

    # Get the three axes
    x_axis = rotation_matrix[:, 0]
    y_axis = rotation_matrix[:, 1]
    z_axis = rotation_matrix[:, 2]

    # Compare alignment with camera's Z-axis
    camera_z = np.array([0, 0, 1])
    alignments = [abs(np.dot(camera_z, x_axis)),
                  abs(np.dot(camera_z, y_axis)),
                  abs(np.dot(camera_z, z_axis))]

    # Find the index of the most aligned axis
    max_index = np.argmax(alignments)

    # Reorder columns to place the most aligned axis as the new Z-axis
    if max_index == 0:  # X-axis is most aligned
        new_rotation = np.column_stack((y_axis, z_axis, x_axis))
    elif max_index == 1:  # Y-axis is most aligned
        new_rotation = np.column_stack((z_axis, x_axis, y_axis))
    else:  # Z-axis is already most aligned
        new_rotation = rotation_matrix  # No change needed

    # Construct the new transformation matrix
    new_transformation = np.eye(4)
    new_transformation[:3, :3] = new_rotation
    new_transformation[:3, 3] = transformation_matrix[:3, 3]


    # Ensure the z-component (row 3, column 3 in new_transformation) is positive
    if new_transformation[2, 2] < 0:  # Check third row, third column
        new_transformation[:, 2] *= -1  # Negate the entire third column

    z_vector = new_transformation[:3, -2:]

    return z_vector
