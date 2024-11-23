import numpy as np
from scipy.io import loadmat
import cv2
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_worldcoords(tag_size=0.152, space_between_tags=0.152, altered_space=0.178, grid_size=(12, 9)):
    world_points = []
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            x_offset = i * (tag_size + space_between_tags) + (altered_space - space_between_tags) * (i // 3)
            y_offset = j * (tag_size + space_between_tags)
            world_points.append([x_offset, y_offset, 0])  # Bottom left corner
            world_points.append([x_offset + tag_size, y_offset, 0])  # Bottom right corner
            world_points.append([x_offset + tag_size, y_offset + tag_size, 0])  # Top right corner
            world_points.append([x_offset, y_offset + tag_size, 0])  # Top left corner
    return np.array(world_points)

def rotation_vector_to_euler(rotation_vector):
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    sy = math.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] +  rotation_matrix[1, 0] * rotation_matrix[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = 0
    return np.array([x, y, z])

def estimate_pose(data, camera_matrix, dist_coeffs, world_points):
    if len(data['id']) == 0:
        print("Skipping this data entry as no tags are identified.")
        return None, None  # Skip this data entry

    image_points = []
    world_points_matched = []

    for i, tag_id in enumerate(data['id']):
        for corner_index, corner in enumerate(['p1', 'p2', 'p3', 'p4']):
            image_point = data[corner][:, i]  # Accessing the corresponding corner for each tag
            image_points.append(image_point)
            world_point_index = int(tag_id) * 4 + corner_index
            world_points_matched.append(world_points[world_point_index])
            print(world_points_matched)

    if len(image_points) < 4 or len(world_points_matched) < 4:
        print("Not enough points for pose estimation.")
        return None, None

    image_points = np.array(image_points, dtype='float32').reshape(-1, 2)  # Reshaping to ensure proper format
    world_points_matched = np.array(world_points_matched, dtype='float32').reshape(-1, 3)

    success, rotation_vector, translation_vector = cv2.solvePnP(world_points_matched, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if success:
        position = translation_vector.flatten()
        orientation = rotation_vector_to_euler(rotation_vector)
        return position, orientation
    else:
        print("Pose estimation was not successful.")
        return None, None
    
# def plot_trajectory(estimated_positions, ground_truth_positions):
#     fig = plt.figure(figsize=(8, 8))
#     ax = fig.add_subplot(111, projection='3d')

#     # Plotting the estimated trajectory
#     ax.plot(estimated_positions[:, 0], estimated_positions[:, 1], estimated_positions[:, 2], label='Estimated', color='blue')

#     # Plotting the ground truth trajectory
#     ax.plot(ground_truth_positions[:, 0], ground_truth_positions[:, 1], ground_truth_positions[:, 2], label='Ground Truth', color='red')

#     ax.set_xlabel('X position')
#     ax.set_ylabel('Y position')
#     ax.set_zlabel('Z position')
#     ax.legend()
#     plt.title('3D Trajectory of the Drone')
#     plt.show()

# def plot_orientation(estimated_orientations, ground_truth_orientations):
#     fig, axs = plt.subplots(3, 1, figsize=(10, 8))

#     # Plotting roll
#     axs[0].plot(estimated_orientations[:, 0], label='Estimated', color='blue')
#     axs[0].plot(ground_truth_orientations[:, 0], label='Ground Truth', color='red')
#     axs[0].set_title('Roll')
#     axs[0].legend()

#     # Plotting pitch
#     axs[1].plot(estimated_orientations[:, 1], label='Estimated', color='blue')
#     axs[1].plot(ground_truth_orientations[:, 1], label='Ground Truth', color='red')
#     axs[1].set_title('Pitch')
#     axs[1].legend()

#     # Plotting yaw
#     axs[2].plot(estimated_orientations[:, 2], label='Estimated', color='blue')
#     axs[2].plot(ground_truth_orientations[:, 2], label='Ground Truth', color='red')
#     axs[2].set_title('Yaw')
#     axs[2].legend()

#     plt.tight_layout()
#     plt.show()
    

# f_x = 314.1779
# f_y = 314.2218 # Focal length in y
# c_x = 199.4848  # Principal point x-coordinate
# c_y = 113.7838  # Principal point y-coordinate

# camera_matrix = np.array([[f_x, 0, c_x],
#                           [0, f_y, c_y],
#                           [0, 0, 1]])

# # If you have distortion coefficients in your parameters.txt
# dist_coeffs = np.array([-0.438607, 0.248625, 0.00072, -0.000476, -0.0911])  # Replace with actual distortion coefficients

# data = loadmat('data\\data\\studentdata1.mat', simplify_cells=True)['data']

# mat_data = loadmat('data\\data\\studentdata1.mat', simplify_cells=True)
# ground_truth_positions = mat_data['vicon'][:, :3]  # Assuming the first three columns are x, y, z positions
# ground_truth_orientations = mat_data['vicon'][:, 3:6]  # Assuming the next three columns are roll, pitch, yaw
# world_points = get_worldcoords()
# # Estimate pose for each data element

# estimated_positions = []
# estimated_orientations = []
# for d in data:
#     position, orientation = estimate_pose(d, camera_matrix, dist_coeffs, world_points)
#     if position is not None and orientation is not None:
#         estimated_positions.append(position)
#         estimated_orientations.append(orientation)

# # Convert lists to numpy arrays for easier handling
# estimated_positions = np.array(estimated_positions)
# estimated_orientations = np.array(estimated_orientations)

# plot_trajectory(estimated_positions, ground_truth_positions)
# plot_orientation(estimated_orientations, ground_truth_orientations)

def visualize_trajectory(mat_file_name, camera_matrix, dist_coeffs, world_points):
    
    mat_data = loadmat(mat_file_name, simplify_cells=True)
    ground_truth_positions = []
    ground_truth_orientations = []
    elements = mat_data['time']
    for idx, _ in enumerate(elements):
        gt = []
        gt_or = []
        x = mat_data['vicon'][0][idx]
        gt.append(x)
        y = mat_data['vicon'][1][idx]
        gt.append(y)
        z = mat_data['vicon'][2][idx]
        gt.append(z)
        ground_truth_positions.append(gt)
        r = mat_data['vicon'][3][idx]
        gt_or.append(r)
        p = mat_data['vicon'][4][idx]
        gt_or.append(p)
        y = mat_data['vicon'][5][idx]
        gt_or.append(y)
        ground_truth_orientations.append(gt_or)


    # ground_truth_positions = mat_data['vicon'][:, :3]
    print(ground_truth_positions)
    # ground_truth_orientations = mat_data['vicon'][:, 3:6]
    print(f"Number of ground truth points: {len(ground_truth_positions)}")
    estimated_positions = []
    estimated_orientations = []

    for data in mat_data['data']:
        position, orientation = estimate_pose(data, camera_matrix, dist_coeffs, world_points)
        if position is not None and orientation is not None:
            estimated_positions.append(position)
            estimated_orientations.append(orientation)

    # Convert to numpy arrays for consistency
    estimated_positions = np.array(estimated_positions)
    estimated_orientations = np.array(estimated_orientations)
    ground_truth_positions = np.array(ground_truth_positions)
    ground_truth_orientations = np.array(ground_truth_orientations)

    print(f"Unique estimated positions: {len(np.unique(estimated_positions, axis=0))}")
    print("Estimated X range:", np.min(estimated_positions[:, 0]), "to", np.max(estimated_positions[:, 0]))
    print("Estimated Y range:", np.min(estimated_positions[:, 1]), "to", np.max(estimated_positions[:, 1]))
    print("Estimated Z range:", np.min(estimated_positions[:, 2]), "to", np.max(estimated_positions[:, 2]))

    fig = plt.figure(figsize=(12, 6))

    # Trajectory scatter plot
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter3D(estimated_positions[:, 0], estimated_positions[:, 1], estimated_positions[:, 2], label='Estimated', color='blue', s=50)
    ax.scatter3D(ground_truth_positions[:, 0], ground_truth_positions[:, 1], ground_truth_positions[:, 2], label='Ground Truth', color='red', s=50)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Trajectory')
    ax.legend()

    # Set axes limits to handle outliers
    ax.set_xlim([np.percentile(estimated_positions[:, 0], 1), np.percentile(estimated_positions[:, 0], 99)])
    ax.set_ylim([np.percentile(estimated_positions[:, 1], 1), np.percentile(estimated_positions[:, 1], 99)])
    ax.set_zlim([np.percentile(estimated_positions[:, 2], 1), np.percentile(estimated_positions[:, 2], 99)])

    # Orientation plots
    labels = ['Roll', 'Pitch', 'Yaw']
    for i in range(3):
        ax = fig.add_subplot(322 + 2 * i)
        ax.plot(estimated_orientations[:, i], label='Estimated', color='blue')
        ax.plot(ground_truth_orientations[:, i], label='Ground Truth', color='red')
        ax.set_title(labels[i])
        ax.legend()

    plt.tight_layout()
    plt.show()

# Usage example
mat_file_name = 'data\\data\\studentdata1.mat'
f_x = 314.1779
f_y = 314.2218 # Focal length in y
c_x = 199.4848  # Principal point x-coordinate
c_y = 113.7838  # Principal point y-coordinate

camera_matrix = np.array([[f_x, 0, c_x],
                          [0, f_y, c_y],
                          [0, 0, 1]])

# If you have distortion coefficients in your parameters.txt
dist_coeffs = np.array([-0.438607, 0.248625, 0.00072, -0.000476, -0.0911])  # Replace with actual distortion coefficients

world_points = get_worldcoords()
print("world points",world_points)
visualize_trajectory(mat_file_name, camera_matrix, dist_coeffs, world_points)