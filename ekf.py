# import cv2
# import numpy as np
# from scipy.io import loadmat
# import math
# # Function to compute the world coordinates of AprilTag corners
# def compute_apriltag_world_coordinates(tag_size, space_between_tags, grid_size):
#     """
#     Compute the world coordinates for the corners of each AprilTag in the grid.
    
#     :param tag_size: The size of each tag (same unit as used in the project, e.g., meters).
#     :param space_between_tags: The space between adjacent tags (same unit as used in the project).
#     :param grid_size: A tuple (num_rows, num_cols) representing the number of tags in each dimension.
#     :return: A dictionary with tag IDs as keys and 4x3 numpy arrays as values, representing the corners' coordinates.
#     """
#     world_coordinates = {}
#     tag_id = 0
#     for row in range(grid_size[0]):
#         for col in range(grid_size[1]):
#             x_offset = col * (tag_size + space_between_tags)
#             y_offset = row * (tag_size + space_between_tags)
#             # Assuming z = 0 for all points since they are on a flat surface
#             corners = np.array([
#                 [x_offset, y_offset, 0],  # Bottom left
#                 [x_offset + tag_size, y_offset, 0],  # Bottom right
#                 [x_offset + tag_size, y_offset + tag_size, 0],  # Top right
#                 [x_offset, y_offset + tag_size, 0]  # Top left
#             ])
#             world_coordinates[tag_id] = corners
#             tag_id += 1
#     return world_coordinates

# # Function to estimate the pose from a single data point
# def estimate_pose(data, camera_matrix, dist_coeffs, tag_world_coordinates):
#     image_points = []
#     object_points = []
    
#     # Collect corresponding object and image points
#     for tag_id, corners in zip(data['id'], [data['p1'], data['p2'], data['p3'], data['p4']]):
#         if tag_id in tag_world_coordinates:
#             object_points.extend(tag_world_coordinates[tag_id])
#             image_points.extend(corners)
    
#     image_points = np.array(image_points, dtype="double")
#     object_points = np.array(object_points, dtype="double")
    
#     # Check if there are enough correspondences
#     if len(image_points) < 4 or len(object_points) < 4:
#         print(f"Not enough points to estimate pose: {len(image_points)} image points, {len(object_points)} object points.")
#         return None, None

#     # Solve PnP
#     success, rotation_vector, translation_vector = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
    
#     if not success:
#         print("solvePnP failed to find a solution.")
#         return None, None

#     # Convert rotation vector to rotation matrix and then to Euler angles
#     rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
#     euler_angles = rotationMatrixToEulerAngles(rotation_matrix)
    
#     return translation_vector, euler_angles


# # Helper function to convert a rotation matrix to Euler angles
# def rotationMatrixToEulerAngles(R):
#     """
#     Calculates rotation matrix to euler angles.

#     :param R: Rotation matrix.
#     :return: Euler angles.
#     """
#     sy = math.sqrt(R[0, 0] * R[0, 0] +  R[1, 0] * R[1, 0])
#     singular = sy < 1e-6

#     if not singular:
#         x = math.atan2(R[2, 1], R[2, 2])
#         y = math.atan2(-R[2, 0], sy)
#         z = math.atan2(R[1, 0], R[0, 0])
#     else:
#         x = math.atan2(-R[1, 2], R[1, 1])
#         y = math.atan2(-R[2, 0], sy)
#         z = 0

#     return np.array([x, y, z])

# # Load data and calibration parameters (adjust paths and names as per your data structure)
# data = loadmat('data\\data\\studentdata5.mat', simplify_cells=True)['data']
# f_x = 314.1779
# f_y = 314.2218 # Focal length in y
# c_x = 199.4848  # Principal point x-coordinate
# c_y = 113.7838  # Principal point y-coordinate

# camera_matrix = np.array([[f_x, 0, c_x],
#                           [0, f_y, c_y],
#                           [0, 0, 1]])

# # If you have distortion coefficients in your parameters.txt
# dist_coeffs = np.array([-0.438607, 0.248625, 0.00072, -0.000476, -0.0911])
# # camera_matrix = np.array(...)  # Load or define your camera matrix here
# # dist_coeffs = np.array(...)  # Load or define your distortion coefficients here
# tag_world_coordinates = compute_apriltag_world_coordinates(tag_size=0.152, space_between_tags=0.152, grid_size=(12, 9))

# # Example usage for a single data point
# single_data_point = data[0]  # Assuming data is a list of data points
# estimated_position, estimated_orientation = estimate_pose(single_data_point, camera_matrix, dist_coeffs, tag_world_coordinates)




import cv2
import numpy as np
import math

# Function to convert rotation matrix to Euler angles
def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] +  R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

# Function to estimate pose given a data point and AprilTag world coordinates
def estimate_pose(data, camera_matrix, dist_coeffs, tag_world_coordinates):
    image_points = []
    object_points = []
    
    # Collect corresponding object and image points
    for i, tag_id in enumerate(data['id']):
        if tag_id in tag_world_coordinates:
            object_points.append(tag_world_coordinates[tag_id].reshape(-1, 3))  # Reshape to ensure correct dimensions
            # Append the image points (corners of the detected tag)
            image_points.append(np.array([data['p1'][i], data['p2'][i], data['p3'][i], data['p4'][i]]).reshape(-1, 2))
    
    # If there are not enough points, return None
    if len(image_points) < 4 or len(object_points) < 4:
        print("Not enough points to estimate pose.")
        return None, None
    
    image_points = np.concatenate(image_points, axis=0)
    object_points = np.concatenate(object_points, axis=0)
    
    # Solve PnP
    success, rotation_vector, translation_vector = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    
    if not success:
        print("solvePnP failed to find a solution.")
        return None, None

    # Convert rotation vector to rotation matrix and then to Euler angles
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    euler_angles = rotationMatrixToEulerAngles(rotation_matrix)
    
    return translation_vector.flatten(), euler_angles

# Sample usage
camera_matrix = np.array(...)  # Define your camera matrix
dist_coeffs = np.array(...)  # Define your distortion coefficients
tag_world_coordinates = ...  # Define your world coordinates for each AprilTag
data = loa
for data_point in data:
    if 'id' in data_point and len(data_point['id']) > 0:
        estimated_position, estimated_orientation = estimate_pose(data_point, camera_matrix, dist_coeffs, tag_world_coordinates)
        if estimated_position is not None and estimated_orientation is not None:
            print(f"Estimated Position: {estimated_position}, Estimated Orientation: {estimated_orientation}")
        else:
            print("Pose estimation failed for this data point.")
    else:
        print("No AprilTags detected in this data point.")
