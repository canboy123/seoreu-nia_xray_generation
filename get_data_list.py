import pydicom
import numpy as np
import time
import cv2
import time
from pathlib import Path
import json
import glob, os
from PIL import Image
import re

########################################
#            INITIALIZATION            #
########################################
saved_dataset_path = '/home/jiacang/datasets/nia_generation/data'
dataset_path = '/home/taehoon/NIA_x-ray_v3/fix'
dcm_dataset_path = f'{dataset_path}/dcm'
json_dataset_path = f'{dataset_path}/json'
saved_path = f'{saved_dataset_path}/output'
Path(f"{saved_path}").mkdir(parents=True, exist_ok=True)

save_as_npy = True
# save_as_npy = False



def main():
    p = Path(json_dataset_path).glob('**/*')
    json_list = [x for x in p if x.is_file()]
    output_lists = []

    count = 0
    # Read the dicom files from the directory
    for tmp_img_file in json_list:
        img_file = str(tmp_img_file)
        try:
            img_id = img_file.split("/")[-1].split(".")
            img_id = ".".join(img_id[:-1])

            # We only concern about these 3 classes data for now
            # if "1.3.12.2.1107.5.3.49.22547.11.201308202032460500.1140989267" not in img_file: continue
            # if "air_fluid_level" not in img_file and "ihps" not in img_file and "normal" not in img_file: continue
            # if "normal" not in img_file : continue
            if "ihps" not in img_file : continue
            count += 1
            print(f"{count} FILE NAME: {img_file}")
            output_lists.append(img_id)
        except Exception as e:
            print(e)
            exit()
    ###########################################
    #               SAVING DATA               #
    ###########################################
    if save_as_npy and len(output_lists) > 0:
        # Save the caption
        caption_filename = f"""ihps_data_lists.txt"""
        caption_file = os.path.join(saved_path, caption_filename)

        with open(caption_file, 'w') as f1:
            for line in output_lists:
                f1.write(f"{line}\n")

    print(f"TOTAL FILE IN {dataset_path}: {count}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()