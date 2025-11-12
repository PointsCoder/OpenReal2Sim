# import os

# def search_tiff_in_py_files(root_dir):
#     tiff_occurrences = []

#     for dirpath, _, filenames in os.walk(root_dir):
#         for filename in filenames:
#             if filename.endswith(".py"):
#                 file_path = os.path.join(dirpath, filename)
#                 try:
#                     with open(file_path, "r", encoding="utf-8") as file:
#                         for i, line in enumerate(file, start=1):
#                             if "tiff" in line.lower():
#                                 tiff_occurrences.append((file_path, i, line.strip()))
#                 except Exception as e:
#                     print(f"Could not read {file_path}: {e}")

#     if tiff_occurrences:
#         print("\nFound 'tiff' mentions:")
#         for path, line_num, content in tiff_occurrences:
#             print(f"{path}:{line_num}: {content}")
#     else:
#         print("No mention of 'tiff' found in any Python files.")

# if __name__ == "__main__":
#     root_directory = "."  # change to your target folder path
#     search_tiff_in_py_files(root_directory)


import os

def find_files_with_background(root_dir):
    matches = []

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if "background" in filename.lower():
                matches.append(os.path.join(dirpath, filename))

    if matches:
        print("\nFiles containing 'background' in their name:")
        for path in matches:
            print(path)
    else:
        print("No files found with 'background' in the name.")

if __name__ == "__main__":
    root_directory = "."  # change to your folder path
    find_files_with_background(root_directory)
