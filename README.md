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
- `read_data.py` is an example script to read the data after the `generate_img_data_from_dcm.py` has been executed.
- `get_data_list.py` is used to retrieve the list of the file under a specific class. The followings txt files are generated from this script.
  - `ihps_data_lists.txt` is the list of the file under `ihps` class.
  - `normal_data_lists.txt` is the list of the file under `normal` class.

---

# Training & Inferencing

## Prerequisite 
- GLIGEN repository
- Install all modules as listed in the `GLIGEN/requirements.txt`.

## Usage
1. Clone the [GLIGEN](https://github.com/gligen/GLIGEN/tree/master?tab=readme-ov-file).
2. Follow the instruction from GLIGEN to make sure the `gligen_inference.py` can run it correctly.
3. You might face the transformer `position_ids` error. This can be solved by lowering down the transformer module version as listed in the `GLIGEN/requirements.txt`.
4. You also might need to download `sd-v1-4.ckpt` online and put it in the `GLIGEN/DATA` directory.
5. Copy and paste the `GLIGEN` from this repository into your GLIGEN repository.
6. The `configs/nia_sem.yaml` is a config file to refer which `train_dataset_names` to be used. 
Since we have used our own dataset and the script is created in the `dataset/custom_datset_sem_dcm.py` and it contains the class `NIASemantic`, 
so we named the `train_dataset_names` as `NIASemantic`.
7. The `dataset/custom_datset_sem_dcm.py` is the dataset class that will read the data from a directory. 
There are 3 files for each datum which are `.npx`, `.npy`, and `.txt` files.
8. The `dataset/catalog.py` is used to point the correct `target` with the proper dataset class (`SemanticDataset` from the `dataset/custom_datset_sem_dcm.py`) 
and the training data directory.
    ```
    self.NIASemantic = {
            "target": "dataset.custom_dataset_sem_dcm.SemanticDataset",
            "train_params":dict(
                dataset_path = os.path.join("/home/jiacang/datasets",'nia_generation/normal'),
            ),
        }
    ```
9. In order to start the training, you can perform `python main.py --name=sem --yaml_file=configs/nia_sem.yaml --gpu_index=0` to train your model.
10. After the model is trained, you can execute `sr_final_gligen_inference_sem.py --gpu_index=1` with a correct checkpoints to generate the artificial images.



















