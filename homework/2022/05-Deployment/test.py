import os

# Check the current working directory
print("Current working directory:", os.getcwd())

# Check if the file exists in the current working directory
file_path = '.\dv.bin'

print("File exists:", os.path.exists(file_path))