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
saved_path = f'{saved_dataset_path}/air_fluid_level'
Path(f"{saved_path}").mkdir(parents=True, exist_ok=True)

save_as_json = True
save_as_json = False

save_as_npy = True
# save_as_npy = False

accepted_labels = ["full", "abdomen", "letter", "testis_shield", "pneumoperitoneum", "ihps", "air_fluid_level", "constipation"]

# Exclude the related clinic_info from the generated caption
exclude_keys = [
    "acquisition_date", "patientid", "height", "wt_group"
]

convert_to_proper_text = {
    "luq의 aggregated bowel gas는 colon과 small bowel gas가 overlap 되어 보이는 것으로 판단됨.": "luq's aggregated bowel gas is judged to be the colon and small bowel gas overlapping.",
    "small bowel dilatation은 보이지 않음.": "small bowel dilatation is not seen.",
    "ruq에 약 1cm 크기의 radiopaque material이 있으며 gb stone 가능성을 배제할 수 없겠음.": "there is radiopaque material about 1 cm in size in ruq, and the possibility of it being gb stone cannot be ruled out.",
    "clinical correlation 및 us 검사를 recommend함.": "we recommend clinical correlation and US testing.",
    "abdominal cavity내 bowel dilatation 및 air-fluid level이 관찰되며, mechanical obstruction가능성이 고려됨.": "bowel dilatation and air-fluid level in the abdominal cavity are observed, and the possibility of mechanical obstruction is considered.",
    "2016.8.17, 18일 abdomen erect supine을 함께 판독함": "2016.8.17, 18th abdomen erect supine was read together",
    "이는 17일 plain radiography와 비교시 호전되었음.": "this was an improvement compared to plain radiography on the 17th.",
    "stomach gas가 prominent 하게 늘어나 있음.": "the stomach gas is prominent.",
    "abnormal calcified and soft tissue density는 보이지 않는다.": "abnormal calcified and soft tissue density is not visible.",
    "일 자": "date",
    "검사명": "inspection name",
    "\n": "<newlineseq>",
    "_x000d_": "<returnseq>",
    "specific bowel gas pattern은 관찰되지 않는다": "no specific bowel gas pattern is observed",
    "====== [conclusion] ======": "conclusion",
    "=========conclusion=========": "conclusion",
    "[conclusion]": "conclusion",
}

def get_status_from_dcm(file_path):
    """
    To get the patient position/orientation from the dicom file
    :param string file_path: The full path of the dicom file
    :return list: patient_position, patient_orientation, view_position
    """
    patient_position, patient_orientation, view_position = "", "", ""
    ds = pydicom.dcmread(file_path)
    if "PatientPosition" in ds:
        patient_position = ds.PatientPosition
        patient_position = patient_position if isinstance(patient_position, str) else ",".join(patient_position)
    if "PatientOrientation" in ds:
        patient_orientation = ds.PatientOrientation
        patient_orientation = patient_orientation if isinstance(patient_orientation, str) else ",".join(patient_orientation)
    if "ViewPosition" in ds:
        view_position = ds.ViewPosition
        view_position = view_position if isinstance(view_position, str) else ",".join(view_position)

    return patient_position, patient_orientation, view_position

def check_contradiction(patient_position, patient_orientation, view_position):
    this_is_contradiction = False
    saved_status = []
    if "supine" in patient_position and "erect" in patient_position:
        saved_status.append("supine erect")
    elif "supine" in patient_position:
        saved_status.append("supine")
    elif "erect" in patient_position:
        saved_status.append("erect")

    if "supine" in patient_orientation and "erect" in patient_orientation:
        saved_status.append("supine erect")
    elif "supine" in patient_orientation:
        saved_status.append("supine")
    elif "erect" in patient_orientation:
        saved_status.append("erect")

    if "supine" in view_position and "erect" in view_position:
        saved_status.append("supine erect")
    elif "supine" in view_position:
        saved_status.append("supine")
    elif "erect" in view_position:
        saved_status.append("erect")

    status = ""
    if len(saved_status) > 0:
        status = saved_status[0]
        for element in saved_status:
            if element != saved_status[0]:
                this_is_contradiction = True
                break

    return this_is_contradiction, status


def main():
    p = Path(json_dataset_path).glob('**/*')
    json_list = [x for x in p if x.is_file()]
    type_check = {}
    age_check = {}
    gender_check = {}
    weight_check = {}
    label_check = {}
    description_check = {}
    status_check = {}
    patient_check = {}
    age_file_count = 0

    count = 0
    # Read the dicom files from the directory
    for tmp_img_file in json_list:
        img_file = str(tmp_img_file)
        # try:
        if True:
            img_id = img_file.split("/")[-1].split(".")
            img_id = ".".join(img_id[:-1])

            class_name = img_file.split("/")[6]

            # We only concern about these 3 classes data for now
            # if "1.3.12.2.1107.5.3.49.22547.11.201308202032460500.1140989267" not in img_file: continue
            # if "air_fluid_level" not in img_file and "ihps" not in img_file and "normal" not in img_file: continue
            if "air_fluid_level" not in img_file : continue
            print(f"{count+1} FILE NAME: {img_file}")

            dcm_file = img_file.replace('json', "dcm")
            patient_position, patient_orientation, view_position = "", "", ""
            if os.path.isfile(dcm_file):
                patient_position, patient_orientation, view_position = get_status_from_dcm(dcm_file)


            if os.path.isfile(img_file):
                if patient_check.get(img_id) is None:
                    patient_check[img_id] = {}

                retrieve_data = {}
                captions = []

                # Make sure the image json (for clinical info and labelling) exists.
                json_file = img_file
                if os.path.isfile(json_file):
                    count += 1

                    with open(json_file, 'r') as f:
                        json_data = json.load(f)

                        # Define the points
                        finding = json_data["shapes"] if json_data.get("shapes") is not None else {}
                        saved_labels = []
                        for info in finding:
                            # If it is a list instead of dictionary, then it mostly the pneumoperitoneum/air_fluid_level/constipation label
                            if isinstance(info, list):
                                info = info[0] if len(info) > 0 else {}
                                label = info["label"] if info.get("label") is not None else ""
                                points = info["potins"] if info.get("potins") is not None else []
                            else:
                                label = info["label"] if info.get("label") is not None else ""
                                points = info["points"] if info.get("points") is not None else []

                            for tmp_label in accepted_labels:
                                if tmp_label in label:
                                    if len(points) >= 2:
                                        if retrieve_data.get(tmp_label) is None:
                                            retrieve_data[tmp_label] = {}

                                        retrieve_data[tmp_label]["points"] = points

                                        if tmp_label.lower() != "full" and tmp_label.lower() != "abdomen" and tmp_label.lower() not in saved_labels:
                                            saved_labels.append(tmp_label.lower())
                                        break

                            if label_check.get(label.lower()) is None:
                                label_check[label.lower()] = 1

                        if len(saved_labels) > 0:
                            captions.append(f"label: {', '.join(saved_labels)}")

                        ###########################################
                        #              CLINICAL DATA              #
                        ###########################################
                        clinic_info = json_data["clinic_info"] if json_data.get("clinic_info") is not None else {}
                        # print(clinic_info)
                        # TODO: Check if the type of the disease is pneumo
                        is_pneumo = False
                        if len(clinic_info.keys()) > 0:
                            for key, val in clinic_info.items():
                                if key.lower() in exclude_keys:
                                    continue

                                tmp_val = val
                                if key.lower() == "gender":
                                    if tmp_val is not None:
                                        if tmp_val.strip().lower() == "m" or tmp_val.strip().lower() == "남":
                                            tmp_val = "male"
                                        if tmp_val.strip().lower() == "f" or tmp_val.strip().lower() == "여":
                                            tmp_val = "female"

                                        if tmp_val is not None and gender_check.get(tmp_val.lower()) is None:
                                            gender_check[tmp_val.lower()] = 0
                                        gender_check[tmp_val.lower()] += 1

                                # Modify the age value to a proper value
                                if key.lower() == "age":
                                    age_file_count += 1
                                    # print(type(val), val is None, val)
                                    if val is not None:
                                        # Replace the korean character to a proper english character
                                        tmp_val = str(tmp_val).lower()
                                        tmp_val = tmp_val.replace("세", "y")
                                        tmp_val = tmp_val.replace("개월", "m")
                                        # Check if the last letter is a character or number

                                        # If it is a digit, we add `y`
                                        if tmp_val[-1].isdigit():
                                            tmp_val = str(tmp_val) + 'y'
                                        if age_check.get(tmp_val.lower()) is None:

                                            age_check[tmp_val.lower()] = 0

                                        age_check[str(tmp_val).lower()] += 1
                                    else:
                                        if age_check.get("-") is None:
                                            age_check["-"] = 0
                                        age_check["-"] += 1

                                if key.lower() == "body_weight":
                                    tmp_val = str(tmp_val).lower()
                                    if weight_check.get(tmp_val) is None:
                                        weight_check[tmp_val] = 0

                                    weight_check[tmp_val] += 1

                                if key.lower() == "description":
                                    if val is not None and description_check.get(val.lower()) is None:
                                        # Replace the korean character to a proper english character
                                        tmp_val = tmp_val.lower()
                                        for search_key, replace_val in convert_to_proper_text.items():
                                            tmp_val = tmp_val.replace(search_key, replace_val)

                                        description_check[tmp_val.lower()] = 1

                                if tmp_val is not None:
                                    if isinstance(tmp_val, str):
                                        tmp_val = tmp_val.lower()
                                    if key.lower() == "type":
                                        if tmp_val == "airfluid":
                                            tmp_val = 'air_fluid_level'
                                        captions.append(f"""diagnosis: {tmp_val}""")
                                    elif key.lower() == "status":
                                        if "air_fluid_level" in img_file:
                                            captions.append(f"""{key.lower()}: erect""")
                                        else:
                                            is_contradiction, tmp_status = check_contradiction(patient_position, patient_orientation, view_position)
                                            if "supine" in tmp_val:
                                                if not is_contradiction:
                                                    captions.append(f"""{key.lower()}: supine""")
                                                else:
                                                    print(f"""{patient_position}, {patient_orientation}, {view_position}, supine, {img_file}""")
                                            elif "erect" in tmp_val:
                                                if not is_contradiction:
                                                    captions.append(f"""{key.lower()}: erect""")
                                                else:
                                                    print(f"""{patient_position}, {patient_orientation}, {view_position}, erect, {img_file}""")
                                            else:
                                                if not is_contradiction and "supine" in tmp_status:
                                                    captions.append(f"""{key.lower()}: supine""")
                                                if not is_contradiction and  "erect" in tmp_status:
                                                    captions.append(f"""{key.lower()}: erect""")

                                    else:
                                        captions.append(f"""{key.lower()}: {tmp_val}""")

                                if "type" in key.lower():
                                    if type_check.get(val.lower()) is None:
                                        type_check[val.lower()] = 1

                                # All air fluid level patient should be erect in the position/status
                                if "status" in key.lower() and "air_fluid_level" in img_file:
                                    patient_check[img_id]["status"] = "erect"
                                    if status_check.get(val.lower()) is None:
                                        status_check[val.lower()] = 0
                                    status_check[val.lower()] += 1
                        else:
                            # Check if the captions has the `diagnosis` caption. Sometimes, the data has no clinical info
                            captions.append(f"""diagnosis: {class_name}""")


                        if patient_check[img_id].get("status") is None:
                            patient_check[img_id]["status"] = ""

                            if status_check.get("-") is None:
                                status_check["-"] = 0
                            status_check["-"] += 1

                        ###########################################
                        #               SAVING DATA               #
                        ###########################################
                        if save_as_npy and len(captions) > 0:
                            # Save the caption
                            caption_filename = f"""{img_id}.txt"""
                            caption_file = os.path.join(saved_path, caption_filename)

                            with open(caption_file, 'w') as f1:
                                for line in captions:
                                    f1.write(f"{line}\n")

                    # exit()
                    # if count >= 200:
                    #     exit()

        # except Exception as e:
        #     print(e)
        #     failed_reading.append(img_file)
        #     exit()

    weights = weight_check.keys()
    tmp_weights = []
    for p in weights:
        if "<" in p or ">" in p:
            continue
            tmp_weights.append(p)
        else:
            tmp_weights.append(float(p))
    weights = tmp_weights
    weights = sorted(weights)
    new_weight_check = {}
    for weight in weights:
        if new_weight_check.get(str(weight)) is None:
            new_weight_check[str(weight)] = 0

            if weight_check.get(str(weight)) is not None:
                new_weight_check[str(weight)] = weight_check[str(weight)]



    print(f"TOTAL FILE IN {dataset_path}: {count}")
    # print(f"TYPES FOUND: { type_check.keys() }")
    # print(f"AGE FOUND: { age_check.items() }")
    print(f"WEIGHT FOUND: { new_weight_check.items() }")
    # print(f"GENDER FOUND: { gender_check.keys() }")
    # print(f"LABELS FOUND: { label_check.keys() }")
    # print(f"STATUS FOUND: { status_check.items() }")
    # print(f"DESCRIPTION FOUND: { description_check.items() }")
    # print(f"AGE FILE COUNT FOUND: { age_file_count }")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()