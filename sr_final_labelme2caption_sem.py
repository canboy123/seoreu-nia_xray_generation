# Labelme2Caption 🚀
# by seareale(Haejin Lee)
# DATE: 2023-07-14

import concurrent.futures
import json
from pathlib import Path
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import pydicom

class labelme2caption:
    ROOT = "./org"
    DEST = "./raw"

    CLASS_NAME_LIST = [
        'normal',
        'pneumoperitoneum',
        'air-fluid_level',
        'constipation',
        'ihps',
    ]

    CLASS_NAME_LIST_KOR = [
        "정상",
        "기복증",
        "공기-액체음영",
        "변비",
        "유문협착증"
    ]

# DIR_NAME = ["D00", "D01", "D04", "D06",]

    POSE = [
        "supine",
        "erect"
    ]

    CLINIC_INFO_KEYS = [
        "Gender",
        "Body_weight",
        "Age",
        "Status",
        "Description",
    ]

    def __init__(self, root=None, dest=None, classes=None):
        self.failed = []
        if root:
            labelme2caption.ROOT = root
        if dest:
            labelme2caption.DEST = dest
        if classes:
            labelme2caption.CLASS_NAME_LIST = classes

    def image_preprocess(self, pix_arr, json_file, imgsz=512):
        ## crop
        crop = pix_arr.copy()
        
        with open(json_file, "r") as js:
            cur_json = json.load(js)
            
        for s in cur_json["shapes"]:
            ## inverse
            # if s["label"] == "04_reverse":
            #     color_4095 = np.full_like(np.zeros(crop.size).reshape(crop.shape), 4095)
            #     crop = color_4095 - crop

            ## crop
            # abdomen bbox 기준으로 crop
            if s["label"] == "02_abdomen":
                points = s["points"]
                x1, y1 = map(int, np.min(points, axis=0))
                x2, y2 = map(int, np.max(points, axis=0))
                x1_pad = x1 - (x2 - x1) * 0.1
                x2_pad = x2 + (x2 - x1) * 0.1
                y1_pad = y1 - (y2 - y1) * 0.1
                y2_pad = y2 + (y2 - y1) * 0.1

                x1_pad, x2_pad = np.clip([x1_pad, x2_pad], 0, crop.shape[1] - 1)
                y1_pad, y2_pad = np.clip([y1_pad, y2_pad], 0, crop.shape[0] - 1)
                crop = np.array(crop)[int(y1_pad):int(y2_pad), int(x1_pad):int(x2_pad)]
            
        ## padding
        h, w = crop.shape[:2]
        max_ = max([h, w])
        cond = h < w
        padding = np.zeros((max_,max_)).astype(np.float64)
        start_ = int(abs(h - w)/2)

        if cond:
            padding[start_:start_+h,:] = crop
        else:    
            padding[:,start_:start_+w] = crop

        three_ch = np.stack([padding] * 3, axis=-1)

        ## resize
        resize = cv2.resize(three_ch, (imgsz,imgsz))
        # resize = Image.fromarray(resize.astype(np.uint8))

        return resize
        
    def convertJson(self, json_file, sem_path, imgsz=512):
        # load json file
        with open(json_file, "r") as js:
            cur_json = json.load(js)
            
        label = Image.fromarray(np.zeros((cur_json["imageHeight"], cur_json["imageWidth"]), dtype=np.uint8))
        label_num = self.CLASS_NAME_LIST.index(json_file.parent.name) + 1

        is_letter = False
        # bbox crop
        no_abdomen = True
        for s in cur_json["shapes"]:
            if s["label"] == "02_abdomen":
                no_abdomen = False
                points = s["points"]
                x1, y1 = map(int, np.min(points, axis=0))
                x2, y2 = map(int, np.max(points, axis=0))

                polygon = []  
                polygon.append((x1, y1))
                polygon.append((x1, y2))
                polygon.append((x2, y2))
                polygon.append((x2, y1))

                draw = ImageDraw.Draw(label)
                draw.polygon(polygon, fill=(label_num))

                x1_pad = x1 - (x2 - x1) * 0.1
                x2_pad = x2 + (x2 - x1) * 0.1
                y1_pad = y1 - (y2 - y1) * 0.1
                y2_pad = y2 + (y2 - y1) * 0.1

                x1_pad, x2_pad = np.clip([x1_pad, x2_pad], 0, np.array(label).shape[1] - 1)
                y1_pad, y2_pad = np.clip([y1_pad, y2_pad], 0, np.array(label).shape[0] - 1)
        
        if no_abdomen:
            label = Image.fromarray(np.full_like(np.array(label), label_num))
            x1_pad, x2_pad = 0, cur_json["imageWidth"]
            y1_pad, y2_pad = 0, cur_json["imageHeight"]
        

        for s in cur_json["shapes"]:
            if s["label"] == "letter":
                points = s["points"]

                x1, y1 = map(int, np.min(points, axis=0))
                x2, y2 = map(int, np.max(points, axis=0))

                polygon = []
                polygon.append((x1, y1))
                polygon.append((x2, y1))
                polygon.append((x2, y2))
                polygon.append((x1, y2))

                draw = ImageDraw.Draw(label)
                draw.polygon(polygon, fill=(6))

        crop = np.array(label)[int(y1_pad):int(y2_pad), int(x1_pad):int(x2_pad)].copy()

        ## padding
        h, w = crop.shape
        max_ = max([h, w])
        cond = h < w
        padding = np.zeros((max_,max_)).astype(np.uint8)
        start_ = int(abs(h - w)/2)

        if cond:
            padding[start_:start_+h,:] = crop
        else:    
            padding[:,start_:start_+w] = crop
            
        ## resize
        resize = cv2.resize(padding, (imgsz,imgsz))
        
        # # 시각화 코드
        # import matplotlib.pyplot as plt
        # print(json_file)
        # plt.figure(figsize=(10, 10))
        # plt.imshow(resize)
        # plt.colorbar()
        # plt.title('Segmentation Mask Visualization')
        # plt.savefig('segmentation.png')
        # exit()

        Image.fromarray(resize).save(sem_path)

        return True

    def get_caption(self, json_file, discription=False):
        # load json file
        with open(json_file, "r") as js:
            cur_json = json.load(js)
        
        disease = Path(json_file).parent.stem
        caption = f'Abdominal X-ray image '
        
        is_clinic_info = "clinic_info" in list(cur_json.keys())
        if is_clinic_info:
            caption += f"of a patient with {disease}, "
            anno = cur_json["clinic_info"]
            for k, v in anno.items():
                
                # # erect만 사용        
                # if k.lower() == 'status' and v.lower() != 'abdomen erect':
                #     return
                
                if discription:
                    clinic_info_keys = self.CLINIC_INFO_KEYS
                else:
                    clinic_info_keys = self.CLINIC_INFO_KEYS[:-1]
                    
                if k in clinic_info_keys and v:
                    if k == 'Age':
                        v = v.replace('세', ' years')
                        v = v.replace('개월', ' months')
                        v = v.replace('일', ' days')
                        
                    caption += str(v).replace('.', '')
                    caption += ' '
                    caption += str(k)
                    caption += ', '
                    
                # # only under age 5
                # if k == 'Age':
                #     if int(v) >= 5:
                #         return
                    
        else: 
            caption += f'with {disease}, '

        for s in cur_json["shapes"]:
            label = s["label"].split("_")
            if label[0] != '01' and label[0] != '02'and label[0] != '04' and label[0] != 'letter':
                caption += str(label[-1])
        
        return caption
    
    def get_pixel_array(self, file_path): 
        ds = pydicom.read_file(file_path)
        pixel_array = ds.pixel_array.copy()

        ## pixel value 최대값이 rwin - lwin이 되도록 조정
        if('WindowCenter' in ds):
            if(type(ds.WindowCenter) == pydicom.multival.MultiValue):
                window_center = float(ds.WindowCenter[0])
                window_width = float(ds.WindowWidth[0])
                lwin = window_center - (window_width / 2.0)
                rwin = window_center + (window_width / 2.0)
            else:    
                window_center = float(ds.WindowCenter)
                window_width = float(ds.WindowWidth)
                lwin = window_center - (window_width / 2.0)
                rwin = window_center + (window_width / 2.0)
        else:
            lwin = np.min(pixel_array)
            rwin = np.max(pixel_array)
        ##
        if(ds.PhotometricInterpretation == 'MONOCHROME1'):
            pixel_array[np.where(pixel_array < lwin)] = lwin
            pixel_array[np.where(pixel_array > rwin)] = rwin
            pixel_array = pixel_array - lwin
            pixel_array = 1.0 - pixel_array
        else:
            pixel_array[np.where(pixel_array < lwin)] = lwin
            pixel_array[np.where(pixel_array > rwin)] = rwin
            pixel_array = pixel_array - lwin
        ##
            
        return pixel_array
    
    def thread_run(self, pbar, dcm_set, root, dest, task):

        for n, dcm_path in enumerate(dcm_set):
            # print(img_path.parent.parts[len(Path(root).parts):])
            dcm_dest_path = Path(dest) / Path(*(dcm_path.parent.parts[len(Path(root).parts):])) / (dcm_path.stem + '.npy')
            dcm_dest_path.parent.mkdir(parents=True, exist_ok=True)
            # print(img_dest_path)
            ## image preprocess
            if task == "img":
                json_file = dcm_path.parent / (dcm_path.stem + ".json")
                if os.path.isfile(dcm_dest_path): continue

                pix_arr = self.get_pixel_array(dcm_path)
                
                try:
                    result_arr = self.image_preprocess(pix_arr, json_file)

                except Exception as e:
                    print(e)
                    raise
                
                np.save(str(dcm_dest_path), result_arr)
                
            ## get bbox info
            if task == "sem": 
                json_file = dcm_path.parent / (dcm_path.stem + ".json")
                sem_dest_path = dcm_dest_path.parent / (dcm_dest_path.stem + ".png")
                if os.path.isfile(sem_dest_path): continue

                if json_file.exists():
                    bbox_results = self.convertJson(json_file, sem_dest_path)               
                else:
                    sem_dest_path.parent.mkdir(parents=True, exist_ok=True)
                    pass
            
            ## get caption
            if task == "caption": 
                json_file = dcm_path.parent / (dcm_path.stem + ".json")
                caption_path = dcm_dest_path.parent / (dcm_dest_path.stem + "_caption.txt")
                if os.path.isfile(caption_path): continue

                caption = self.get_caption(json_file, discription=False)
                
                # discription key 사용
                # caption = self.get_caption(json_file, discription=True)
                
                if caption is None:
                    continue
                caption_path.parent.mkdir(parents=True, exist_ok=True)
                # caption = 'Abdominal x-ray image of a patient without disease'
                with open(str(caption_path), "w") as f:
                    f.write(caption)

            pbar.update(1)

    def run(self, task="img"):
        dcm_list = sorted(list(Path(labelme2caption.ROOT).rglob("*.dcm")))
        '''
        for jpg_f in jpg:
            correct = False
            for pk_f in pk:
                if pk_f.stem == jpg_f.stem:
                    correct = True
            if not correct:
                img_list.append(jpg_f)
        '''
        print(f"{labelme2caption.ROOT}: {len(dcm_list)}")

        array_split_list = np.array_split(dcm_list, 1)
        
        with tqdm(dcm_list) as pbar:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = {
                    executor.submit(self.thread_run, pbar, array_set, 
                                    labelme2caption.ROOT, 
                                    labelme2caption.DEST, task): array_set for array_set in array_split_list
                }
            for future in concurrent.futures.as_completed(results):
                try:
                    future.result()
                except Exception as e:
                    import traceback
                    print(traceback.format_exc())
                    print(e)
