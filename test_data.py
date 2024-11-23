import numpy as np
import cv2
from scipy.io import loadmat

def get_worldcoords():
    rows = 12
    cols = 9
    tag_size = 0.152  # size of each tag
    normal_spacing = 0.152  # normal spacing between tags
    adjusted_spacing = 0.178  # adjusted spacing between certain columns

    # Tag map in column-major order (as provided)
    tag_map = np.array([
        [0, 12, 24, 36, 48, 60, 72, 84, 96],
        [1, 13, 25, 37, 49, 61, 73, 85, 97],
        [2, 14, 26, 38, 50, 62, 74, 86, 98],
        [3, 15, 27, 39, 51, 63, 75, 87, 99],
        [4, 16, 28, 40, 52, 64, 76, 88, 100],
        [5, 17, 29, 41, 53, 65, 77, 89, 101],
        [6, 18, 30, 42, 54, 66, 78, 90, 102],
        [7, 19, 31, 43, 55, 67, 79, 91, 103],
        [8, 20, 32, 44, 56, 68, 80, 92, 104],
        [9, 21, 33, 45, 57, 69, 81, 93, 105],
        [10, 22, 34, 46, 58, 70, 82, 94, 106],
        [11, 23, 35, 47, 59, 71, 83, 95, 107]
    ])

    # Initialize dictionary to hold world coordinates keyed by tag ID
    world_coords = {}

    for col in range(cols):
        for row in range(rows):
            # Adjust y-coordinate based on column index due to adjusted spacing
            if col < 3:
                y = col * (tag_size + normal_spacing)
            elif col < 6:
                y = 3 * (tag_size + normal_spacing) + (col - 3) * (tag_size + adjusted_spacing)
            else:
                y = 3 * (tag_size + normal_spacing) + 3 * (tag_size + adjusted_spacing) + (col - 6) * (tag_size + normal_spacing)

            x = row * (tag_size + normal_spacing)
            tag_id = tag_map[row, col]
            world_coords[tag_id] = (x, y)

    return world_coords

def estimate_pose(data, camera_matrix, dist_coeffs):
    # Get world coordinates of all tags
    world_coords = get_worldcoords()
    
    # Initialize arrays to store pose results
    estimated_positions = []
    estimated_orientations = []

    # Process each image/frame in the dataset
    for packet in data['data']:
        if 'id' in packet and packet['id'].size > 0:
            object_points = []
            image_points = []

            # Collect corresponding world and image points for observed tags
            for i, tag_id in enumerate(packet['id']):
                if tag_id in world_coords:
                    # Append world coordinates (3D points)
                    object_points.append(world_coords[tag_id] + (0,))  # z-coordinate is 0
                    # Append image coordinates (2D points)
                    image_points.append([packet['p1'][:, i], packet['p2'][:, i], packet['p3'][:, i], packet['p4'][:, i]])

            object_points = np.array(object_points).reshape(-1, 3).astype(np.float32)
            image_points = np.array(image_points).reshape(-1, 2).astype(np.float32)

            if len(object_points) >= 4:  # Ensure enough points for pose estimation
                # Solve for pose
                success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)

                if success:
                    # Convert rotation vector to rotation matrix
                    R, _ = cv2.Rodrigues(rvec)
                    # Convert rotation matrix to Euler angles
                    euler_angles = rotation_matrix_to_euler_angles(R)

                    estimated_positions.append(tvec.flatten())
                    estimated_orientations.append(euler_angles)

    return estimated_positions, estimated_orientations

def rotation_matrix_to_euler_angles(R):
    # This function converts a rotation matrix to Euler angles
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])

# Example usage:
# Assuming `camera_params`, `data`, `camera_matrix`, and `dist_coeffs` are preloaded
camera_params = loadmat('path_to_your_parameters_file.mat', simplify_cells=True)
camera_matrix = camera_params['camera_matrix']
dist_coeffs = camera_params['dist_coeffs']
data = loadmat('path_to_your_data_file.mat', simplify_cells=True)

positions, orientations = estimate_pose(data, camera_matrix, dist_coeffs)
print("Estimated Positions:", positions)
print("Estimated Orientations:", orientations)


def estimate_pose(data, camera_matrix, distortion_coefficients, tags_info):
    if len(data['id']) == 0:
        print("Skipping this data entry as no tags are identified.")
        return None, None  # Skip this data entry

    image_points = []
    world_points_matched = []

    # Loop over each detected tag in the current frame of data
    for i, tag_id in enumerate(data['id']):
        for corner_index, corner in enumerate(['p1', 'p2', 'p3', 'p4']):
            image_point = data[corner][:, i]  # Accessing the corresponding corner for each tag
            image_points.append(image_point)
            world_point_index = int(tag_id) * 4 + corner_index
            world_points_matched.append(world_points[world_point_index])
            # print("MAATTCHHEDDDD WORLD",world_points_matched)

    if len(image_points) < 4 or len(world_points_matched) < 4:
        print("Not enough points for pose estimation.")
        return None, None

    image_points = np.array(image_points, dtype='float32').reshape(-1, 2)  # Reshaping to ensure proper format
    world_points_matched = np.array(world_points_matched, dtype='float32').reshape(-1, 3)

    # Solving PnP
    _, rvec, tvec = cv2.solvePnP(
        world_points_matched,
        image_points,
        camera_matrix,
        distortion_coefficients,
        flags=cv2.SOLVEPNP_ITERATIVE
    )


    # Rodrigues to convert rotation vector to matrix
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    # Combine rotation matrix and translation vector to form camera to world frame matrix
    camera_to_world_frame = np.hstack((rotation_matrix, tvec.reshape(-1, 1)))
    camera_to_world_frame = np.vstack((camera_to_world_frame, [0, 0, 0, 1]))

    camera_to_world_frame = np.linalg.inv(camera_to_world_frame)

    # Transformation from camera to drone frame, defined by the rotation and translation
    rotation_z = np.array([
        [np.cos(np.pi / 4), -np.sin(np.pi / 4), 0],
        [np.sin(np.pi / 4), np.cos(np.pi / 4), 0],
        [0, 0, 1]
    ])
    rotation_x = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])
    rotation = rotation_x @ rotation_z
    camera_to_drone_frame = np.array([
        [rotation[0, 0], rotation[0, 1], rotation[0, 2], -0.04],
        [rotation[1, 0], rotation[1, 1], rotation[1, 2], 0],
        [rotation[2, 0], rotation[2, 1], rotation[2, 2], -0.03],
        [0, 0, 0, 1]
    ])

    # Transforming camera frame to drone frame by combining matrices
    drone_to_world_frame = camera_to_world_frame @ camera_to_drone_frame

    # Extract the position and rotation from the transformation matrix
    position = drone_to_world_frame[:3, 3]
    orientation_matrix = drone_to_world_frame[:3, :3]
    orientation = rotation_matrix_to_euler_angles(orientation_matrix)

    return position, orientation