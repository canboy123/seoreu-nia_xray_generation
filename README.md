# NIA X-ray Images Generation

This is a project to generate artificial abdomen x-ray images using [GLIGEN](https://github.com/gligen/GLIGEN/tree/master?tab=readme-ov-file).

---

# Dataset Preprocessing
We need to preprocess the dataset before we can feed into GLIGEN to train the model.
First of all, we extract the pixel array (i.e. the image values) from the dicom (.dcm) files and save  each of them as .npy files.
After that, we generate the mask using the bbox  (i.e., abdomen, letter, testis_shield, pneumoperitoneum, ihps, air_fluid_level and constipation) given from the json files and save each of them as .npx file.
Lastly, we create the caption for each datum and save each of them as .txt file.

## Prerequisite Modules
- pydicom

## Usage
- `generate_img_data_from_dcm.py` is a script to generate the dataset that can be used in the GLIGEN later. 
This process includes the pixel array extraction, mask and caption generation.
Before execute this script, you need to make sure the dataset folder is arranged as follows:
    ```
    .
    └── NIA_x-ray_v3
        └── fix
            ├── dcm     # The folder that contain .dcm files, and followed by the class folder name
            │   ├── air_fluid_level
            │   ├── constipation
            │   ├── ihps
            │   ├── normal
            │   └──  pneumoperitoneum
            └── json     # The folder that contain .json files, and followed by the class folder name
                ├── air_fluid_level
                ├── constipation
                ├── ihps
                ├── normal
                └──  pneumoperitoneum
    ```
  Execute the script with `python generate_img_data_from_dcm.py`.
- `regenerate_caption_only.py` is used to regenerate the caption if the caption is generated incorrectly.
This is used to reduce the time from reading the dicom files while generate the caption as needed.