import os
import glob
import random

def find_obj_files(directory):
    obj_files = []
    # Recursively search for .obj files in the directory and subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.obj'):
                obj_files.append(os.path.join(root, file))
    return obj_files

def get_category_dict(obj_files):
    obj_dict = {}
    for file_path in obj_files:
        key = file_path.split('/')[-4]
        if key in obj_dict.keys():
            obj_dict[key].append(file_path)
        else: 
            obj_dict[key] = [file_path]
    return obj_dict

def get_final_objs(obj_dict, n_cat, n_obj):
    paths = []
    keys = list(obj_dict.keys())[:n_cat]
    for key in keys:
        paths +=obj_dict[key][:n_obj]
    return paths

import os
import shutil

def copy_files_with_index(file_paths, destination_dir):
    # Ensure the destination directory exists
    os.makedirs(destination_dir, exist_ok=True)

    for idx, file_path in enumerate(file_paths):
        new_file_name = f"{idx}.obj"
        destination_path = os.path.join(destination_dir, new_file_name)
        shutil.copy(file_path, destination_path)
        #print(f"Copied {file_path} to {destination_path}")
    

# Example usage
#
#     obj_dict = {}
#     directory_path = "torch-ngp/OmniObj3D"
#     obj_files = find_obj_files(directory_path)
#     print("Found .obj files:")
#     for file_path in obj_files:
#         print(file_path)
#     print(len(obj_files))
#     print(obj_dict.keys())  # Correctly call the keys() method

#     for file_path in obj_files:
#         key = file_path.split('/')[-4]
#         if key in obj_dict.keys():
#             obj_dict[key].append(file_path)
#         else: 
#             obj_dict[key] = [file_path]

#     directory_path = input("Enter the path of the directory to search: ")
#     obj_files = find_obj_files(directory_path)
#     print("Found .obj files:")
#     for file_path in obj_files:
#         print(file_path)
