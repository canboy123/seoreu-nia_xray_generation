# NIA X-ray Images Genration

This is a project to generate artificial abdomen x-ray images using [GLIGEN](https://github.com/gligen/GLIGEN/tree/master?tab=readme-ov-file).

## Usage
- `generate_img_data_from_dcm.py` is a script to generate the dataset that can be used in the GLIGEN later. 
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

## Installation and Build
```bash
# build
> .\build\build.bat
# output in dist folder - `build\captos`

> cd build
# 1) build `captos-setup.exe` using setup.iss
# before this, you need to modify variables in `setup.iss` file
# 2) change `Output\captos.exe` to `Output\captos-setup.exe`
> .\cert.ps1 -build msi
> python hash.py
```

## Convention
```
# TODO: something to do
# NOTE: something to note
# CHECKLIST: description of checklist
# [ ]: undone task in checklist
# [x]: done task in checklist
# HACK: implemented with bad code
# FIXME: I will fix this code
# XXX: you should fix this code
# BUG: there is a bug in this code
```

## PyTest
It is recommended to make sure the `users` and `patient` tables are empty before running the following test.
```commandline
> cd app/viewer-backend/tests/API

# To run the test for the `auth` API
> pytest test_auth.py

# To run the test for the `dashboard` API
> pytest test_dashboard.py

# To run the test for the `viewer` API
> pytest test_viewer.py
```