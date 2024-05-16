from utils_ import *
import os
from SyntheticImage import SyntheticImage
import time

NB_SAMPLES = 250
# Set the paths 
LABEL_PATH = "/home/hghallab/dataset/object_detection_data/real_raw/train/labels"
BACKGROUNDS_PATH = "/home/hghallab/dataset/synthetic_data/backgrounds"
SIGNS_PATH = "/home/hghallab/dataset/synthetic_data/signs/signs that need 250 samples/squares"
OUTPUT_PATH = "/home/hghallab/dataset/synthetic_data/outputs250"

BACKGROUDS = os.listdir(BACKGROUNDS_PATH)
SIGNS = os.listdir(SIGNS_PATH)
LABELS = os.listdir(LABEL_PATH)



tic = time.time()
for folder in SIGNS:
    print("creating data from folder ", folder, "...")
    signs_path = os.path.join(SIGNS_PATH, folder)
    sign = os.listdir(signs_path)
    for i in range(NB_SAMPLES):
        sign_path, background_path = get_random_sign_and_bg(sign, BACKGROUDS, signs_path, BACKGROUNDS_PATH)
        # Define the transform to apply to the final compostion
        transform_comp = A.Compose([
            A.HistogramMatching([background_path],
                                blend_ratio=(0.8, 1),
                                always_apply=True,
                                p=1),
            A.GaussNoise(var_limit=(10, 50), mean=0),
            A.ElasticTransform(alpha=10, sigma=30, alpha_affine=30, p=0.7),
            A.RandomFog(fog_coef_lower=0.01, fog_coef_upper=0.1, p=1),
            A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.5)                                                
        ])
        # Define the transform to apply to the sign
        transforms_obj = A.Compose([
            A.RandomBrightnessContrast(
                                       contrast_limit=0.1,
                                       brightness_limit=(-0.1, 0),
                                       brightness_by_max=True,
                                       always_apply=False,
                                       p=1),
            A.RGBShift(r_shift_limit=(-50, 0), g_shift_limit=(-50, 0), b_shift_limit=(-50, 0), p=1),
            A.GaussianBlur(blur_limit=(3, 5), p=1),
        ])
        synthetic_image = SyntheticImage(background_path, sign_path)
        synthetic_image.get_img_and_mask()
        yolo_coordinates = synthetic_image.retrieve_yolo_coordinates(LABEL_PATH, LABELS)
        x_center, y_center, width, height = synthetic_image.yolo_to_bbox(yolo_coordinates)
        
        bbox = synthetic_image.yolo_to_bbox_(yolo_coordinates)
        synthetic_image.transform_background(bbox)

        synthetic_image.resize_transform_obj(int(height), int(width), transforms=transforms_obj)
        synthetic_image.create_composition(int(x_center), int(y_center), transform_comp)
        synthetic_image.save_composition_and_annotation(OUTPUT_PATH, folder)
tac = time.time()
print("time", tac - tic)