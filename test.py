from scipy.io import loadmat
# import numpy as np
# # data = loadmat('data\\data\\studentdata0.mat', simplify_cells = True)

# # data_array_dicts = data['data']

# # ids = []

# # for data_array in data_array_dicts:
    

# # data = 



# # print(ids)

# def load_data(filename):
#     """
#     Load and parse the .mat file containing the drone data.

#     :param filename: Path to the .mat file.
#     :return: A list of dictionaries, each containing data for a single timestamp.
#     """
#     # Load the .mat file
#     mat_data = loadmat(filename, simplify_cells=True)
    
#     # Initialize a list to hold parsed data
#     parsed_data = []
    
#     # Iterate through each entry in the data struct array
#     for entry in mat_data['data']:
#         # Create a dictionary for each set of measurements
#         print("entry",entry)
#         data_dict = {
#             'img': entry['img'],  # Image data
#             'id': entry['id'],  # AprilTag IDs
#             'corners': [entry['p1'], entry['p2'], entry['p3'], entry['p4']],  # Tag corners
#             'rpy': entry['rpy'],  # Roll, pitch, yaw
#             'omg': entry['omg'],  # Angular velocity
#             'acc': entry['acc'],  # Linear acceleration
#         }
        
#         # Append this dictionary to the list
#         parsed_data.append(data_dict)
#     # print("this is dict",data_dict)
#     return parsed_data

# # Usage example
# data_filename = 'data\\data\\studentdata5.mat'
# data = load_data(data_filename)

# # print("THIS IS THE DATA MAN",data[100]['id'])
# load_data(data_filename)
data = loadmat('data\\data\\studentdata1.mat', simplify_cells=True)
for key in data:
    print(f"Key: {key}, Type: {type(data[key])}, Data: {data[key]}")