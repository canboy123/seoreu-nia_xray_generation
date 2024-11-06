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
import glob, os
from PIL import Image


########################################
#            INITIALIZATION            #
########################################
saved_dataset_path = '/home/jiacang/datasets/nia_generation/data'
dataset_path = '/home/taehoon/NIA_x-ray_v3/fix'
dcm_dataset_path = f'{dataset_path}/dcm'
json_dataset_path = f'{dataset_path}/json'
saved_path = f'{saved_dataset_path}/normal'
Path(f"{saved_path}").mkdir(parents=True, exist_ok=True)

save_as_json = True
save_as_json = False

save_as_npy = True
# save_as_npy = False

save_as_image = True
save_as_image = False

accepted_labels = ["full", "abdomen", "letter", "testis_shield", "pneumoperitoneum", "ihps", "air_fluid_level", "constipation"]

# Exclude the related clinic_info from the generated caption
exclude_keys = [
    "acquisition_date", "patientid", "height", "wt_group"
]

mask_label = {
    "padding": 0,
    "abdomen": 1,
    "plate": 2,
    "pneumoperitoneum": 3,
    "ihps": 4,
    "arifluidlevel": 5,
    "constipation": 6,
    "letter": 7,
}

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

resize_shape = (512, 512)

def save_dictionary(dictionary, file_path):
    with open(file_path, 'w') as file:
        json.dump(dictionary, file)


def load_dictionary(file_path):
    with open(file_path, 'r') as file:
        dictionary = json.load(file)
    return dictionary

def draw_mask(image, mask, color=0.5) :
    """
    Generate the proper segmentation mask for all categories
    :param list image: The segmentation mask that need to be overwrited by the `mask`
    :param list mask:
    :param color:
    :return:
    """
    masked_image = image.copy()
    masked_image = np.where(mask.astype(int), color, masked_image)
    return masked_image

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

def get_pixel_array(file_path):
    """
    To get the proper pixel array from the dicom file
    :param string file_path: The full path of the dicom file
    :return list: Pixel array data
    """
    ds = pydicom.dcmread(file_path)
    # print("ds", ds)
    pixel_array = ds.pixel_array.copy()

    try:
        # Some data might get error by using apply_modality_lut() function
        pixel_array = apply_modality_lut(pixel_array, ds)
    except:
        pass

    if ('WindowCenter' in ds):
        if (type(ds.WindowCenter) == pydicom.multival.MultiValue):
            window_center = float(ds.WindowCenter[0])
            window_width = float(ds.WindowWidth[0])
            min_px = window_center - (window_width / 2.0)
            max_px = window_center + (window_width / 2.0)
        else:
            window_center = float(ds.WindowCenter)
            window_width = float(ds.WindowWidth)
            min_px = window_center - (window_width / 2.0)
            max_px = window_center + (window_width / 2.0)
    else:
        # Get Bits Stored and Bits Allocated from the DICOM metadata
        # bits_stored = ds.BitsStored
        #
        # # The range of possible pixel values
        # min_possible_value = 0
        # max_possible_value = (2 ** bits_stored) - 1
        #
        # min_px = min_possible_value * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
        # max_px = max_possible_value * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
        min_px = np.min(pixel_array)
        max_px = np.max(pixel_array)

    # Normalize the pixel data
    pixel_array = np.clip(pixel_array, min_px, max_px)
    pixel_array = pixel_array.astype(float)
    pixel_array = (pixel_array - min_px) / float(max_px - min_px)

    if (ds.PhotometricInterpretation == 'MONOCHROME1'):
        pixel_array = 1.0 - pixel_array

    return pixel_array

def get_finalized_points(points, ori_img_size, file_img_size, is_exceed_ratio=False):
    """
    To get the finalize points, including the trasformation for the points back to the original ratio and
    the swapping of the first point and the second point if the rectangle is drawn from bottom left to upper right
    :param list points: The original coordinate for the rectangle
    :param tuple ori_img_size: original width, original height
    :param tuple file_img_size: written width, written height
    :param bool is_exceed_ratio: True if the written img size is larger than the original img size, else False
    :return: Transformed coordinate
    """
    _x1, _y1 = points[0]
    _x2, _y2 = points[1]
    ori_w, ori_h = ori_img_size
    json_w, json_h = file_img_size

    # Transform the point back to the original size ratio
    if is_exceed_ratio:
        _x1 = int(np.ceil(_x1 / float(json_w) * ori_w))
        _y1 = int(np.ceil(_y1 / float(json_h) * ori_h))
        _x2 = int(np.ceil(_x2 / float(json_w) * ori_w))
        _y2 = int(np.ceil(_y2 / float(json_h) * ori_h))

    w = int(abs(round(_x1) - round(_x2)))
    h = int(abs(round(_y1) - round(_y2)))

    x1 = int(_x1)
    x2 = int(_x2)
    y1 = int(_y1)
    y2 = int(_y2)
    if _x1 > _x2:
        x1 = int(int(round(_x1)) - w)
        x2 = int(int(round(_x2)) + w)
    if _y1 > _y2:
        y1 = int(int(round(_y1)) - h)
        y2 = int(int(round(_y2)) + h)

    return x1, y1, x2, y2

def get_scaled_points(p1, p2, cropped_p1, cropped_p2, padding, padded_image):
    """
    To get the scaled coordinate after the actual image is resized to a user-defined size (e.g.: (512, 512))
    :param list p1: The first coordinate that usually defined on the top left point
    :param list p2: The second coordinate that usually defined on the bottom right point (just insert as (0, 0) if it is a polygon)
    :param list cropped_p1: The first coordinate of the cropped image (top left point)
    :param list cropped_p2: The second coordinate of the cropped image (bottom right point)
    :param list padding: The padding values that have been added to the cropped image
    :param list padded_image: The image which has added the padding
    :return list: x1, y1, x2, y2
    """
    x1, y1 = p1
    x2, y2 = p2
    cropped_x1, cropped_y1 = cropped_p1
    cropped_x2, cropped_y2 = cropped_p2
    x1 = max(x1 - cropped_x1, 0)
    y1 = max(y1 - cropped_y1, 0)
    x2 = min(x2 - cropped_x1, cropped_x2 - cropped_x1)
    y2 = min(y2 - cropped_y1, cropped_y2 - cropped_y1)

    # Then, adjust for padding
    x1 += padding[0]
    x2 += padding[0]
    y1 += padding[1]
    y2 += padding[1]

    # Calculate the scaling factors for width and height
    scale_x = resize_shape[0] / padded_image.shape[1]  # new_width / original_width
    scale_y = resize_shape[1] / padded_image.shape[0]  # new_height / original_height

    # Adjust the label coordinates based on the scaling factors
    x1 = int(np.floor(x1 * scale_x))
    y1 = int(np.floor(y1 * scale_y))
    x2 = int(np.ceil(x2 * scale_x))
    y2 = int(np.ceil(y2 * scale_y))

    return x1, y1, x2, y2

def get_polygon_mask(points, resized_image):
    # Initialize a blank mask
    mask = np.zeros(resized_image.shape, dtype=np.uint8)
    # Draw the filled polygon on the mask
    mask = cv2.fillPoly(mask, [points], color=255)

    return mask

def main():
    p = Path(dcm_dataset_path).glob('**/*')
    print(dcm_dataset_path)
    dcm_list = [x for x in p if x.is_file()]
    type_check = {}
    age_check = {}
    gender_check = {}
    weight_check = {}
    label_check = {}
    patient_position_check = {}
    patient_orientation_check = {}
    view_position_check = {}
    description_check = {}
    status_check = {}
    patient_check = {}
    age_file_count = 0

    saved_json_data = {}
    captions_json = {}
    failed_reading = []

    count = 0
    # Read the dicom files from the directory
    for tmp_img_file in dcm_list:
        img_file = str(tmp_img_file)
        # try:
        if True:
            # Need to get the class for each file
            tmp = img_file.split(dcm_dataset_path)[-1]

            img_id = img_file.split("/")[-1].split(".")
            img_id = ".".join(img_id[:-1])

            class_name = img_file.split("/")[6]

            # Do not repeat to run the same file if the patient has been processed before the script has error
            img_npy_filename = f"""{img_id}.npy"""
            img_npy_file = os.path.join(saved_path, img_npy_filename)
            if os.path.isfile(img_npy_file):
                print(img_npy_file, os.path.isfile(img_npy_file))
                continue

            # if img_id != "W01_D01_01-2112108687_001": continue

            # We only concern about these 3 classes data for now
            # if "air_fluid_level" not in img_file and "ihps" not in img_file and "normal" not in img_file: continue
            if "normal" not in img_file : continue
            print(f"{count+1} FILE NAME: {img_file}")

            if os.path.isfile(img_file):
                if patient_check.get(img_id) is None:
                    patient_check[img_id] = {}

                if img_file[-4:] == ".dcm":
                    # Get pixel data from the dicom
                    pixel_array = get_pixel_array(img_file)
                    json_file = img_file.replace('dcm', "json")

                    patient_position, patient_orientation, view_position = get_status_from_dcm(img_file)

                    patient_check[img_id]["PP"] = patient_position
                    if patient_position_check.get(patient_position) is None:
                        patient_position_check[patient_position] = 0

                    patient_position_check[patient_position] += 1

                    patient_check[img_id]["PO"] = patient_orientation
                    if patient_orientation_check.get(patient_orientation) is None:
                        patient_orientation_check[patient_orientation] = 0

                    patient_orientation_check[patient_orientation] += 1

                    patient_check[img_id]["VP"] = view_position
                    if view_position_check.get(view_position) is None:
                        view_position_check[view_position] = 0

                    view_position_check[view_position] += 1
                elif img_file[-4:] == ".jpg":
                    pixel_array = cv2.imread(img_file)
                    pixel_array = cv2.cvtColor(pixel_array, cv2.COLOR_BGR2GRAY)
                    # Need to normalize them
                    pixel_array = (pixel_array - np.min(pixel_array)) / (np.max(pixel_array)- np.min(pixel_array))

                    json_file = img_file.replace("dcm", "json")
                    json_file = json_file.replace(".jpg", ".json")

                has_plate = False
                retrieve_data = {}
                img_npy = []
                seg_npy = []
                captions = []

                # Make sure the image json (for clinical info and labelling) exists.
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
                        #       IMAGE AND SEGMENTATION DATA       #
                        ###########################################
                        ori_h, ori_w = pixel_array.shape

                        cropped_pixel_array = pixel_array
                        resized_image = cropped_pixel_array

                        cropped_x1, cropped_y1, cropped_x2, cropped_y2 = 0, 0, ori_w, ori_h

                        full_detail = retrieve_data["full"] if retrieve_data.get("full") is not None else {}
                        abdomen_detail = retrieve_data["abdomen"] if retrieve_data.get("abdomen") is not None else {}
                        # Check if abdomen exists or not. If not, we take the `full` box
                        if len(abdomen_detail.keys()) <= 0:
                            abdomen_detail = full_detail    # Assume full_detail always exists
                        letter_detail = retrieve_data["letter"] if retrieve_data.get("letter") is not None else {}
                        plate_detail = retrieve_data["testis_shield"] if retrieve_data.get("testis_shield") is not None else {}
                        pneumoperitoneum_detail = retrieve_data["pneumoperitoneum"] if retrieve_data.get("pneumoperitoneum") is not None else {}
                        air_fluid_level_detail = retrieve_data["air_fluid_level"] if retrieve_data.get("air_fluid_level") is not None else {}
                        constipation_detail = retrieve_data["constipation"] if retrieve_data.get("constipation") is not None else {}

                        abdomen_mask, letter_mask, plate_mask, pneumoperitoneum_mask, air_fluid_level_mask, constipation_mask = None, None, None, None, None, None

                        # To check if the ratio is exceeding from the original size.
                        # If yes, we need to re-transform the point back to the original size
                        is_exceed_ratio = False
                        json_w, json_h = 0, 0
                        # Check if the json has enlarged from the original ratio
                        if full_detail.get("points") is not None:
                            _x1, _y1 = points[0]
                            _x2, _y2 = points[1]

                            json_w = int(abs(round(_x1) - round(_x2)))
                            json_h = int(abs(round(_y1) - round(_y2)))

                            if ori_w < json_w or ori_h < json_h:
                                is_exceed_ratio = True

                        if abdomen_detail.get("points") is not None:
                            x1, y1, x2, y2 = get_finalized_points(abdomen_detail["points"], (ori_w, ori_h), (json_w, json_h), is_exceed_ratio)

                            # Saved the cropped image coordinate for the future use
                            cropped_x1,  cropped_y1, cropped_x2, cropped_y2 = x1, y1, x2, y2

                            # Step 1: Crop the image based on the bounding box
                            cropped_pixel_array = cropped_pixel_array[y1:y2, x1:x2]
                            cropped_height, cropped_width = cropped_pixel_array.shape

                            abdomen_mask = np.ones(cropped_pixel_array.shape, dtype=np.uint8)

                            # Step 2: Add padding to make the cropped image square
                            # Calculate padding amounts
                            if cropped_height > cropped_width:
                                pad_total = cropped_height - cropped_width
                                pad_left = pad_total // 2
                                pad_right = pad_total - pad_left
                                padded_image = np.pad(cropped_pixel_array, ((0, 0), (pad_left, pad_right)), mode='constant', constant_values=0)
                                abdomen_mask = np.pad(abdomen_mask, ((0, 0), (pad_left, pad_right)), mode='constant', constant_values=0)
                                padding = (pad_left, 0)  # Padding (x, y)
                            else:
                                pad_total = cropped_width - cropped_height
                                pad_top = pad_total // 2
                                pad_bottom = pad_total - pad_top
                                padded_image = np.pad(cropped_pixel_array, ((pad_top, pad_bottom), (0, 0)), mode='constant', constant_values=0)
                                abdomen_mask = np.pad(abdomen_mask, ((pad_top, pad_bottom), (0, 0)), mode='constant', constant_values=0)
                                padding = (0, pad_top)  # Padding (x, y)

                            # Step 3: Resize the padded square image to 512x512
                            resized_image = cv2.resize(padded_image, resize_shape)
                            abdomen_mask = cv2.resize(abdomen_mask, resize_shape)

                        if letter_detail.get("points") is not None:
                            x1, y1, x2, y2 = get_finalized_points(letter_detail["points"], (ori_w, ori_h), (json_w, json_h), is_exceed_ratio)
                            x1, y1, x2, y2 = get_scaled_points((x1, y1), (x2, y2), (cropped_x1, cropped_y1), (cropped_x2, cropped_y2), padding, padded_image)

                            letter_mask = np.zeros((resized_image.shape[0], resized_image.shape[1]), dtype=np.uint8)
                            letter_mask[y1:y2, x1:x2] = 1

                            saved_json_data["letter"] = [[x1, y1], [x2, y2]]

                        if plate_detail.get("points") is not None:
                            x1, y1, x2, y2 = get_finalized_points(plate_detail["points"], (ori_w, ori_h), (json_w, json_h), is_exceed_ratio)
                            x1, y1, x2, y2 = get_scaled_points((x1, y1), (x2, y2), (cropped_x1, cropped_y1), (cropped_x2, cropped_y2), padding, padded_image)

                            plate_mask = np.zeros((resized_image.shape[0], resized_image.shape[1]), dtype=np.uint8)
                            plate_mask[y1:y2, x1:x2] = 1

                            has_plate = True

                        if pneumoperitoneum_detail.get("points") is not None:
                            tmp_points = pneumoperitoneum_detail["points"]
                            points = []
                            for coordinate in tmp_points:
                                x1, y1, _, _ = get_finalized_points([(coordinate[0], coordinate[1]), (ori_w, ori_h)], (ori_w, ori_h), (json_w, json_h), is_exceed_ratio)
                                x1, y1, _, _ = get_scaled_points((x1, y1), (0, 0), (cropped_x1, cropped_y1), (cropped_x2, cropped_y2), padding, padded_image)
                                points.append((x1, y1))
                            points = np.array(points, np.int32)
                            points = points.reshape((-1, 1, 2))

                            pneumoperitoneum_mask = get_polygon_mask(points, resized_image)

                        if air_fluid_level_detail.get("points") is not None:
                            tmp_points = air_fluid_level_detail["points"]
                            points = []
                            for coordinate in tmp_points:
                                x1, y1, _, _ = get_finalized_points([(coordinate[0], coordinate[1]), (ori_w, ori_h)], (ori_w, ori_h), (json_w, json_h), is_exceed_ratio)
                                x1, y1, _, _ = get_scaled_points((x1, y1), (0, 0), (cropped_x1, cropped_y1), (cropped_x2, cropped_y2), padding, padded_image)
                                points.append((x1, y1))
                            points = np.array(points, np.int32)
                            points = points.reshape((-1, 1, 2))

                            air_fluid_level_mask = get_polygon_mask(points, resized_image)

                        if constipation_detail.get("points") is not None:
                            tmp_points = constipation_detail["points"]
                            points = []
                            for coordinate in tmp_points:
                                x1, y1, _, _ = get_finalized_points([(coordinate[0], coordinate[1]), (ori_w, ori_h)], (ori_w, ori_h), (json_w, json_h), is_exceed_ratio)
                                x1, y1, _, _ = get_scaled_points((x1, y1), (0, 0), (cropped_x1, cropped_y1), (cropped_x2, cropped_y2), padding, padded_image)
                                points.append((x1, y1))
                            points = np.array(points, np.int32)
                            points = points.reshape((-1, 1, 2))

                            constipation_mask = get_polygon_mask(points, resized_image)

                        saved_json_data["image"] = resized_image.tolist()
                        img_npy = resized_image.tolist()

                        # Add abdomen mask first
                        merge_mask = draw_mask(resized_image, abdomen_mask, mask_label["abdomen"])
                        # Add plate mask second
                        if plate_mask is not None:
                            merge_mask = draw_mask(merge_mask, plate_mask, mask_label["plate"])
                        if pneumoperitoneum_mask is not None:
                            merge_mask = draw_mask(merge_mask, pneumoperitoneum_mask, mask_label["pneumoperitoneum"])

                        if constipation_mask is not None:
                            merge_mask = draw_mask(merge_mask, constipation_mask, mask_label["constipation"])

                        # Add letter mask last
                        if letter_mask is not None:
                            merge_mask = draw_mask(merge_mask, letter_mask, mask_label["letter"])

                        seg_npy = merge_mask

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
                        if save_as_npy and len(img_npy) > 0:
                            # Save the image values into npy
                            img_npy_filename = f"""{img_id}.npy"""
                            img_npy_file = os.path.join(saved_path, img_npy_filename)

                            with open(img_npy_file, 'wb') as f1:
                                np.save(f1, np.array(img_npy))

                            # Save the segmentation
                            seg_npy_filename = f"""{img_id}.npx"""
                            seg_npy_file = os.path.join(saved_path, seg_npy_filename)

                            with open(seg_npy_file, 'wb') as f1:
                                np.save(f1, np.array(seg_npy))

                            # Save the caption
                            caption_filename = f"""{img_id}.txt"""
                            caption_file = os.path.join(saved_path, caption_filename)

                            with open(caption_file, 'w') as f1:
                                for line in captions:
                                    f1.write(f"{line}\n")

                        if save_as_image:
                            new_name = f"""{img_id}.jpg"""
                            new_name = os.path.join(saved_path, new_name)
                            saved_image = (seg_npy / 7) * 255
                            cv2.imwrite(os.path.join(saved_path, new_name), saved_image)

                            new_name = f"""ori_{img_id}.jpg"""
                            new_name = os.path.join(saved_path, new_name)
                            saved_image = resized_image * 255
                            cv2.imwrite(os.path.join(saved_path, new_name), saved_image)

                    # exit()
                    # if count >= 200:
                    #     exit()

                    # if save_as_json and len(saved_json_data) > 0:
                    #     saved_json_name = img_file.replace('.dcm', ".json")
                    #     saved_json_name = os.path.join(saved_path, saved_json_name)
                    #     save_dictionary(saved_json_data, saved_json_name)

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
    print(f"TYPES FOUND: { type_check.keys() }")
    print(f"AGE FOUND: { age_check.keys() }")
    print(f"GENDER FOUND: { gender_check.keys() }")
    print(f"WEIGHT FOUND: { new_weight_check.items() }")
    print(f"LABELS FOUND: { label_check.keys() }")
    print(f"PATIENT POSITION FOUND: { patient_position_check.items() }")
    print(f"PATIENT ORIENTATION FOUND: { patient_orientation_check.items() }")
    print(f"VIEW POSITION FOUND: { view_position_check.items() }")
    print(f"STATUS FOUND: { status_check.items() }")
    print("Failed reading", failed_reading)

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()