import os
import shutil

def construct_data():
    source = "D:/Data/FEI face/manually_aligned"
    target_neutral = "D:/Data/FEI face/neutral"
    target_smile = "D:/Data/FEI face/smile"

    file_list = os.listdir(source)

    for file in file_list:
        file_path = os.path.join(source, file)
        if "a" in file: #a: neutral
            shutil.copy(file_path, os.path.join(target_neutral, file))
        else:
            shutil.copy(file_path, os.path.join(target_smile, file))


if __name__ == '__main__':
    construct_data()
