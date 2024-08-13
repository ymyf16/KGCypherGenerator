import shutil
import os
import sys

def clear_folder(directory_path):
    # Remove the entire directory
    shutil.rmtree(directory_path)
    # Recreate the empty directory
    os.makedirs(directory_path)

if __name__ == "__main__":
    # Get the directory path from command-line arguments
    if len(sys.argv) != 2:
        print("Usage: python clear_folder.py <directory_path>")
        sys.exit(1)

    folder_path = sys.argv[1]
    clear_folder(folder_path)
