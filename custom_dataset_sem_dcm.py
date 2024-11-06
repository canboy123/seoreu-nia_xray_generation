import os 
import torch 
import json 
from PIL import Image, ImageDraw, ImageOps
import torchvision.transforms as transforms
import random
import torchvision.transforms.functional as TF
import numpy as np
from pathlib import Path


def recursively_read(rootdir, must_contain, exts=["png", "jpg", "JPEG", "jpeg"]):
    out = [] 
    for r, d, f in os.walk(rootdir):
        for file in f:
            if (file.split('.')[1] in exts)  and  (must_contain in os.path.join(r, file)):
                out.append(os.path.join(r, file))
    return out


def exist_in(short_str, list_of_string):
    for string in list_of_string:
        if short_str in string:
            return True 
    return False 


def clean_files(image_files, normal_files):
    """
    Not sure why some images do not have normal map annotations, thus delete these images from list. 

    The implementation here is inefficient .....  
    """
    new_image_files = []

    for image_file in image_files:
        image_file_basename = os.path.basename(image_file).split('.')[0]
        if exist_in(image_file_basename,normal_files):
            new_image_files.append(image_file)
    image_files = new_image_files


    # a sanity check 
    for image_file, normal_file in zip(image_files, normal_files):
        image_file_basename = os.path.basename(image_file).split('.')[0]
        normal_file_basename = os.path.basename(normal_file).split('.')[0]
        assert image_file_basename == normal_file_basename[:-7] 
    
    return image_files, normal_files




class SemanticDataset():
    def __init__(self, dataset_path, prob_use_caption=1, image_size=512, random_flip=False):
        self.dataset_path = dataset_path
        self.prob_use_caption = prob_use_caption 
        self.image_size = image_size
        self.random_flip = random_flip

        
        # Image and normal files 
        arr_files = recursively_read(rootdir=self.dataset_path, must_contain="", exts=['npy'])
        arr_files.sort()
        sem_files = recursively_read(rootdir=self.dataset_path, must_contain="", exts=['png'])
        sem_files.sort()
        caption_files = recursively_read(rootdir=self.dataset_path, must_contain="", exts=['txt'])
        caption_files.sort()
        

        self.arr_files = arr_files
        self.sem_files = sem_files
        self.caption_files = caption_files

        print(len(self.arr_files), len(self.sem_files), len(self.caption_files))
        assert len(self.arr_files) == len(self.sem_files) == len(self.caption_files)
        self.pil_to_tensor = transforms.PILToTensor()


    def total_images(self):
        return len(self)


    def __getitem__(self, index):

        arr_path = self.arr_files[index]
        

        out = {}

        out['id'] = index
        pix_tensor = torch.tensor(np.load(arr_path).astype(np.float32))
        pix_tensor = pix_tensor.permute(2, 0, 1)
        # sem = Image.open(self.sem_files[index]).convert("L") # semantic class index 0,1,2,3,4 in uint8 representation
        sem = np.load(self.sem_files[index]).astype(np.int8) # semantic class index 0,1,2,3,4 in uint8 representation

        # assert image.size == sem.size

        
        # - - - - - center_crop, resize and random_flip - - - - - - #  

        # crop_size = min(pix_arr.shape[:2])
        # pix_arr = TF.center_crop(pix_arr, crop_size)
        # pix_arr = pix_arr.resize( (self.image_size, self.image_size) )

        # sem = TF.center_crop(sem, crop_size)
        # sem = sem.resize( (self.image_size, self.image_size), Image.NEAREST ) # acorrding to official, it is nearest by default, but I don't know why it can prodice new values if not specify explicitly

        # if self.random_flip and random.random()<0.5:
        #     pix_arr = ImageOps.mirror(pix_arr)
        #     sem = ImageOps.mirror(sem)       

        sem = self.pil_to_tensor(sem)[0,:,:]

        input_label = torch.zeros(152, self.image_size, self.image_size)
        sem = input_label.scatter_(0, sem.long().unsqueeze(0), 1.0)

        with open(self.caption_files[index], "r") as f:
            out['caption'] = f.read()
  
        pix_tensor =  pix_tensor - torch.min(pix_tensor) 
        out['image'] = ( ( pix_tensor ) / ( torch.max(pix_tensor) - torch.min(pix_tensor) ) - 0.5 ) / 0.5
        out['sem'] = sem
        out['mask'] = torch.tensor(1.0) 
        


        # -------------------- caption ------------------- # 
        # Open caption json 
        # with open(self.caption_files[index], 'r') as f:
        #     self.image_filename_to_caption_mapping = json.load(f)
            
        # if random.uniform(0, 1) < self.prob_use_caption:
        #     out["caption"] = self.image_filename_to_caption_mapping[ os.path.basename(image_path) ]
        # else:
        #     out["caption"] = ""

        return out


    def __len__(self):
        return len(self.arr_files)


