from pathlib import Path
from sr_final_labelme2caption_sem import labelme2caption

path = Path('directory of image and json files')

# preprocess with label2caption.py
DATA_PATH = str(path)
SAVE_PATH = Path('directory to save data for train')

converter = labelme2caption(root=DATA_PATH, dest=str(SAVE_PATH))
converter.run(task="img")
converter.run(task="sem")
converter.run(task="caption")

print("finally preprocessing is finished")
# check the number of after files
image_list = sorted(list(SAVE_PATH.rglob("*.npy")))
caption_list = sorted(list(SAVE_PATH.rglob("*_caption.txt")))
label_list = sorted(list(SAVE_PATH.rglob("*.png")))

print("# of images:", len(image_list))
print("# of captions:", len(caption_list))
print("# of labels:", len(label_list))