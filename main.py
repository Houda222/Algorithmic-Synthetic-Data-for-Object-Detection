from utils_ import *
import os
from SyntheticImage import SyntheticImage
import time
import albumentations as A

######################## The algorithm
# The idea is to take a background image, which is an already annotated image with only one sign, 
# and put the new sign in the same position of the old sign 
# based on its label file. 
# Some transformations are applied to the sign image to make them more realistic
# and generate different instances
# Some other transformation are applied to the background images to try to remove the old sign. 
# And finally some transformations are applied to the final image to simulate speed (elastic tranform)
# to simulate different weather conditions etc.
########################


######################## Folders and Files
# Sign images are put in folders based on their class
# for example: /synthetic_data/signs/signs that need 200 samples/squares/12
# 12 here signifies the class 12
# One could directly retrieve the label from the label file
########################


# Define the number of generated outputs
NB_SAMPLES = 300

# Set the paths 
LABEL_PATH = "/home/hghallab/dataset/object_detection_data/RealestRealRaw/train/labels"
BACKGROUNDS_PATH = "/home/hghallab/dataset/synthetic_data/backgrounds"
SIGNS_PATH = "/home/hghallab/dataset/synthetic_data/signs/signs that need 200 samples"
OUTPUT_PATH = "/home/hghallab/dataset/synthetic_data/outputs200"

BACKGROUDS = os.listdir(BACKGROUNDS_PATH)
SIGNS = os.listdir(SIGNS_PATH)
LABELS = os.listdir(LABEL_PATH)

# Set a time counter
tic = time.time()

# Generation loop
for folder in SIGNS:
    print("creating data from folder ", folder, "...")

    signs_path = os.path.join(SIGNS_PATH, folder)
    sign = os.listdir(signs_path)
    
    for i in range(NB_SAMPLES):
        # Get a random index for the sign and background
        sign_path, background_path = get_random_sign_and_bg(sign, BACKGROUDS, signs_path, BACKGROUNDS_PATH)
        
        # Define the transform to apply to the final compostion
        transform_comp = A.Compose([
            A.HistogramMatching([background_path],
                                blend_ratio=(0.8, 1),
                                always_apply=True,
                                p=1),
            A.GaussNoise(var_limit=(10, 50), mean=0),
            A.GaussianBlur(blur_limit=(3, 5), p=1),
            A.Perspective(scale=(0.05, 0.1), keep_size=True, p=0.7), 
            A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=3, p=0.5), 
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), shadow_dimension=5, p=0.3), 
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
            A.Rotate(limit=(-10, 10), p=0.5)
        ])
        
        # Initilize the class with given parameters
        synthetic_image = SyntheticImage(background_path, sign_path)

        # Generate the mask
        synthetic_image.get_img_and_mask()

        # Get yolo coordinates from label files
        yolo_coordinates = synthetic_image.retrieve_yolo_coordinates(LABEL_PATH, LABELS)
        
        # Transform coordinates to be able to position the new sign on the background
        x_center, y_center, width, height = synthetic_image.yolo_to_bbox(yolo_coordinates)
        
        # Get bounding boxes of the old sign to remove it from the background
        bbox = synthetic_image.yolo_to_bbox_(yolo_coordinates)
        synthetic_image.transform_background(bbox)

        # Resize and apply tranforms to the sign image
        synthetic_image.resize_transform_obj(int(height), int(width), transforms=transforms_obj)
        
        # Merge everything and get results
        synthetic_image.create_composition(int(x_center), int(y_center), transform_comp)
        
        # Save the output image and its new labels file in the specified direction
        synthetic_image.save_composition_and_annotation(OUTPUT_PATH, folder)

tac = time.time()
print("time", tac - tic)