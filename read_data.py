import pydicom
import numpy as np
import time
import cv2
import numpy as np
import time
from pathlib import Path
import json
from skimage import exposure
import base64
import os
from pydicom.pixels import apply_voi_lut, apply_modality_lut


########################################
#            INITIALIZATION            #
########################################
dataset_path = '/home/jiacang/datasets/nia_generation/data'
saved_path = f'{dataset_path}/train_2'
Path(f"{saved_path}").mkdir(parents=True, exist_ok=True)

save_as_json = True
# save_as_json = False

save_as_image = True
save_as_image = False

my_caption_token_path = f"my_caption_token.json"

exclude_keys = [
    "Acquisition_date", "PatientID"
]

def main():
    saved_json_data = {}
    captions_json = {}
    failed_reading = []
    count = 0

    file = "/home/jiacang/datasets/nia_generation/data/train_2/202408211343413220002.npy"
    with open(file, 'rb') as f:
        img_data = np.load(f)
    print(np.min(img_data), np.max(img_data), img_data.shape)

    file = "/home/jiacang/datasets/nia_generation/data/train_2/202408211343413220002.npx"
    with open(file, 'rb') as f:
        seg_data = np.load(f)
    print(np.min(seg_data), np.max(seg_data), seg_data.shape)

    file = "/home/jiacang/datasets/nia_generation/data/train_2/202408211343413220002.txt"
    with open(file, 'r') as f:
        caption = f.read()

    print("img_data", img_data)
    print("seg_data", seg_data)
    print("caption", caption)


    # saved_image = np.array(data["image"]) * 255
    # print(np.shape(saved_image))
    # cv2.imwrite(os.path.join(saved_path, "xxx.jpg"), saved_image)

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()