import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import albumentations as A
import time
import random
from tqdm import tqdm
import shutil



#################### code for getting data for first GAN training trial #############################

def copy_image_with_class(image_path, label_path, dest_dir, classes):
    """
    Copies an image to the destination directory if its corresponding label file
    contains at least one of the desired classes.

    Args:
        image_path (str): Path to the image file.
        label_path (str): Path to the corresponding label file.
    """

    # Extract image filename (without extension)
    image_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    try:
        # Open the label file in read mode
        with open(label_path, 'r') as f:
            lines = f.readlines()

        # Check if any desired class is present in the label file
        for line in lines:
            class_id, x_min, y_min, x_max, y_max = line.strip().split(' ')
            if int(class_id) in classes:
                # Copy the image to the destination directory
                shutil.copy2(image_path, os.path.join(dest_dir, image_filename + ".jpg"))
                print(f"Copied {image_filename}.jpg to destination directory.")
                return

    except FileNotFoundError:
        print(f"Label file {label_path} not found.")
    except IOError as e:
        print(f"Error processing label file {label_path}: {e}")

def copy_images_with_class(img_source_dir, label_source_dir, dest_dir, classes):
    # Iterate through all files in the source directory
    for filename in os.listdir(img_source_dir):
        # Check if it's an image file
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(img_source_dir, filename)

            # Extract the corresponding label file name
            label_filename = os.path.splitext(filename)[0] + ".txt"
            label_path = os.path.join(label_source_dir, label_filename)

            # Check if the label file exists
            if os.path.exists(label_path):
                copy_image_with_class(image_path, label_path, dest_dir, classes)
            else:
                print(f"Label file {label_path} not found for image {filename}.")

    print("Filtering completed!")


def get_class_index(desired_classes, all_classes:list):
    idxs = []
    for class_name in desired_classes:
        idxs.append(all_classes.index(class_name))
    return idxs
    
def get_random_sign_and_bg(signs, backgrounds, signs_path, backgrounds_path):
    rand_sign = random.randint(0, len(signs)-1)
    rand_bg = random.randint(0, len(backgrounds)-1)

    rand_sign = os.path.join(signs_path, signs[rand_sign])
    rand_bg = os.path.join(backgrounds_path, backgrounds[rand_bg])

    return rand_sign, rand_bg

########################## Some useful functions #########################

def rename_files(folder):
    # Iterate through subfolders in the source directory
    i = 0
    for filename in os.listdir(folder):
        source_path = os.path.join(folder, filename)
        new_filename = str(i)+".png"  # Create a variable to hold the new filename
        destination_path = os.path.join(folder, new_filename)

        # Rename the file using os.rename
        os.rename(source_path, destination_path)
        i += 1
        print(f" {filename} renamed to {new_filename} in {folder}")

    print("Finished")


def delete_images_and_labels(img_source_dir, label_source_dir, classes):
    cpt = 0
    # Iterate through all files in the source directory
    for filename in os.listdir(img_source_dir):
        # Check if it's an image file
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(img_source_dir, filename)

            # Extract the corresponding label file name
            label_filename = os.path.splitext(filename)[0] + ".txt"
            label_path = os.path.join(label_source_dir, label_filename)

            if check_class(label_path, classes):
                cpt += 1
                print(image_path)
                print(label_path)
                os.remove(image_path)
                os.remove(label_path)

    print(f"deleted {cpt} images and {cpt} label files")

def check_class(label_path, classes):
    with open(label_path, "r") as file:
        cls = file.readline()[:2]
        for cls_ in classes:
            if cls == cls_:
                return True
            


delete_images_and_labels("/home/hghallab/dataset/synthetic_data/outputs300/images", "/home/hghallab/dataset/synthetic_data/outputs300/labels", ["60"])
