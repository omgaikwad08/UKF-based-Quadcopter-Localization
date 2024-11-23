import cv2
import numpy as np
from scipy.io import loadmat

def load_matfile(data):
    data0 = loadmat('data\\data\\studentdata0.mat', simplify_cells=True)
    data1 = loadmat('data\\data\\studentdata1.mat', simplify_cells=True)
    data2 = loadmat('data\\data\\studentdata2.mat', simplify_cells=True)
    data3 = loadmat('data\\data\\studentdata3.mat', simplify_cells=True)
    data4 = loadmat('data\\data\\studentdata4.mat', simplify_cells=True)
    data5 = loadmat('data\\data\\studentdata5.mat', simplify_cells=True)
    data6 = loadmat('data\\data\\studentdata6.mat', simplify_cells=True)
    data7 = loadmat('data\\data\\studentdata7.mat', simplify_cells=True)

def world_coords(tag_ids):
    tag_width = 0.152
    tag_spacing = 0.152
    diff_spacing = 0.178

    world_coordinates = []

    for tag_id in tag_ids:
        row = tag_id//9
        col = tag_id%9

        x = row * (tag_width + tag_spacing)
        # Adjust for the additional spacing between certain columns
        y = col * (tag_width + tag_spacing) + (diff_spacing - tag_spacing) * (col > 2) + (diff_spacing - tag_spacing) * (col > 5)

        corners = [
            (x, y, 0),  # bottom left
            (x, y + tag_width, 0),  # bottom right
            (x + tag_width, y + tag_width, 0),  # top right
            (x + tag_width, y, 0)  # top left
        ]

        world_coordinates.append(corners)

        return world_coordinates  

def rotation_matrix_to_euler_angles(R):
    """
    Convert a rotation matrix to Z-X-Y Euler angles (yaw, pitch, roll).
    
    :param R: A 3x3 rotation matrix.
    :return: A tuple containing the Euler angles (yaw, pitch, roll).
    """
    assert R.shape == (3, 3), "R must be a 3x3 matrix"

    # Pitch (theta)
    # Use arcsin but also ensure the value is within [-1, 1] to avoid numerical issues
    theta = np.arcsin(max(min(-R[2, 0], 1.0), -1.0))

    # Check for gimbal lock
    if np.abs(R[2, 0]) < 0.99999:
        # Roll (phi) and Yaw (psi)
        phi = np.arctan2(R[2, 1], R[2, 2])
        psi = np.arctan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock situation - set roll to 0 and determine yaw directly
        phi = 0
        if R[2, 0] < 0:
            psi = np.arctan2(R[0, 1], R[0, 2])
        else:
            psi = np.arctan2(-R[0, 1], -R[0, 2])

    return psi, theta, phi



def estimate_pose(data, camera_matrix, coeffs):
    data_array_dicts = data['data']

   
    for data_array in data_array_dicts:
        

        tag_ids = data_array['id']
        # data_array = data['data']
        time = data_array['t']
        corners = [data_array['p1'], data_array['p2'], data_array['p3'], data_array['p4']]

    world_frame_pts = world_coords(tag_ids)
    image_points = np.array(corners).reshape(-1, 2)  # Reshape the corners array to the format expected by solvePnP
    # Solve the PnP problem
    _, rotation_vector, translation_vector = cv2.solvePnP(world_frame_pts, image_points, camera_matrix, coeffs)
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    euler_angles = rotation_matrix_to_euler_angles(rotation_matrix)

    return translation_vector, euler_angles

f_x = 314.1779
f_y = 314.2218 # Focal length in y
c_x = 199.4848  # Principal point x-coordinate
c_y = 113.7838  # Principal point y-coordinate

camera_matrix = np.array([[f_x, 0, c_x],
                          [0, f_y, c_y],
                          [0, 0, 1]])

# If you have distortion coefficients in your parameters.txt
dist_coeffs = np.array([-0.438607, 0.248625, 0.00072, -0.000476, -0.0911])  # Replace with actual distortion coefficients

data = loadmat('data\\data\\studentdata0.mat', simplify_cells = True)


estimate_pose(data, camera_matrix=camera_matrix, coeffs=dist_coeffs)

