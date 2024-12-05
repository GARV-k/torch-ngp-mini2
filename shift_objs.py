import os
import math

def shift_vertices_x(input_folder, output_folder, shift_amount=10):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Loop over all .obj files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".obj"):
            input_file_path = os.path.join(input_folder, filename)
            output_file_path = os.path.join(output_folder, filename)
            
            with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
                for line in input_file:
                    if line.startswith('v '):  # Look for vertex lines
                        parts = line.split()
                        x = float(parts[1]) + shift_amount  # Shift x coordinate
                        new_line = f"v {x} {parts[2]} {parts[3]}\n"
                        output_file.write(new_line)
                    else:
                        output_file.write(line)  # Copy other lines unchanged

    print(f"Shifted .obj files have been saved to: {output_folder}")
    
    
def rotate_vertices_y(input_folder, output_folder, angle_degrees=90):
    # Convert the rotation angle from degrees to radians
    angle_radians = math.radians(angle_degrees)
    cos_angle = math.cos(angle_radians)
    sin_angle = math.sin(angle_radians)
    
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Loop over all .obj files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".obj"):
            input_file_path = os.path.join(input_folder, filename)
            output_file_path = os.path.join(output_folder, filename)
            
            with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
                for line in input_file:
                    if line.startswith('v '):  # Look for vertex lines
                        parts = line.split()
                        x = float(parts[1])
                        z = float(parts[3])
                        
                        # Rotate around Y-axis
                        new_x = x * cos_angle - z * sin_angle
                        new_z = x * sin_angle + z * cos_angle
                        
                        # Write the updated vertex line
                        new_line = f"v {new_x} {parts[2]} {new_z}\n"
                        output_file.write(new_line)
                    else:
                        output_file.write(line)  # Copy other lines unchanged

    print(f"Rotated .obj files have been saved to: {output_folder}")

# Example usage
folder_path = "datasets/for_2_objs/"  # Replace with your folder path
# for idx in [10,20,30,40,50,60]:
for idx in [90]:
    output_folder_path = f"datasets/rot_after_inf/"  # Replace with your output folder path
    rotate_vertices_y(folder_path, output_folder_path,180)
# import os

# # Specify the directory containing the .obj files
# directory = output_folder_path

# # Iterate over all files in the directory
# for filename in os.listdir(directory):
#     if filename.endswith('.obj'):
#         # Split the filename to extract the part after the last underscore
#         parts = filename.split('_')
#         new_filename = f"{parts[-1]}"
        
#         # Get full file paths
#         old_filepath = os.path.join(directory, filename)
#         new_filepath = os.path.join(directory, new_filename)
        
#         # Rename the file
#         os.rename(old_filepath, new_filepath)

# print("Renaming completed.")