import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import albumentations as A
import time
import random
from tqdm import tqdm
import shutil


########################### Functions to make basic synthetic data #########################




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
    
# img_source_dir = "/home/hghallab/dataset/real_raw/train/images"
# label_source_dir = "/home/hghallab/dataset/real_raw/train/labels"
# dest_dir = "/home/hghallab/dataset/GANs_dataset/FirstGan"

# names = ['1 feu violet 2 feux horizontaux', '2 feux jaunes chassis R', 'Avertissement - Ralentissement 30 and Chassis H -6 - 3 feux en S-', 'Avertissement - rappel 30 and Chassis H -6 - 3 feux en S-', 'Avertissement -1 feu jaune- and Chassis A -3 feux verticaux-', 'Avertissement -1 feu jaune- and Chassis C -5 feux verticaux-', 'Avertissement -1 feu jaune- and Chassis F -6 - 1 feux en L inverse-', 'Avertissement -1 feu jaune- and Chassis H -6 - 3 feux en S-', 'Avertissement -1 feu jaune- and Chassis R -6 feux dans un disque-', 'Carre -2 feux rouges- and Chassis C -5 feux verticaux-', 'Carre -2 feux rouges- and Chassis F -6 - 1 feux en L inverse-', 'Carre -2 feux rouges- and Chassis H -6 - 3 feux en S-', 'Carre violet -1 feu violet- and Chassis C -5 feux verticaux-', 'F', 'Feu blanc -1 feu blanc- and Chassis A -3 feux verticaux-', 'Feu blanc -1 feu blanc- and Chassis ID2 -2 feux horizontaux-', 'Feu blanc -1 feu blanc- and Chassis ID3 -3 feux horizontaux-', 'Feu vert -1 feu vert fixe- and Chassis A -3 feux verticaux-', 'Feu vert -1 feu vert fixe- and Chassis C -5 feux verticaux-', 'Feu vert -1 feu vert fixe- and Chassis F -6 - 1 feux en L inverse-', 'Feu vert -1 feu vert fixe- and Chassis H -6 - 3 feux en S-', 'Feux blancs -2 feux blancs- and Chassis ID2 -2 feux horizontaux-', 'Feux blancs -2 feux blancs- and Chassis ID3 -3 feux horizontaux-', 'G', 'Nf', 'Off -No light- and Chassis A -3 feux verticaux-', 'Off -No light- and Chassis C -5 feux verticaux-', 'Off -No light- and Chassis F -6 - 1 feux en L inverse-', 'Off -No light- and Chassis H -6 - 3 feux en S-', 'Off -No light- and Chassis ID2 -2 feux horizontaux-', 'Off -No light- and Chassis ID3 -3 feux horizontaux-', 'Off -No light- and Chassis R -6 feux dans un disque-', 'R', 'Ralentissement 30 -2 feux jaunes horizontaux- and Chassis F -6 - 1 feux en L inverse-', 'Ralentissement 30 -2 feux jaunes horizontaux- and Chassis H -6 - 3 feux en S-', 'Rappel 30 -2 feux jaunes verticales- and Chassis H -6 - 3 feux en S-', 'Semaphore -1 feu rouge- and Chassis A -3 feux verticaux-', 'Semaphore -1 feu rouge- and Chassis C -5 feux verticaux-', 'Semaphore -1 feu rouge- and Chassis F -6 - 1 feux en L inverse-', 'Semaphore -1 feu rouge- and Chassis H -6 - 3 feux en S-', 'Z', 'baissez panto distance', 'baissez panto execution', 'baissez panto fin', 'carre blanc', 'carre noir', 'carre violet', 'carre violet -1 feu violet- and -2 feux verticaux-', 'cercle blanc', 'cercle jaune', 'cercle noir', 'chevron pointe bas', 'chevron pointe haut', 'coupez courant', 'coupez courant fin', 'demi cercle blanc', 'demi cercle inv', 'disque rouge mecanique', 'drapeau jaune bleu', 'ecran', 'feu blanc -1 feu blanc- and chassis ID2 -2feux verticaux-', 'feux', 'fin de catenaire', 'fin de parcours', 'fleche jaune bleue', 'jalon arret', 'jalon manoeuvre', 'losange blanc', 'losange jaune avertissement mecanique', 'mirliton', 'mirliton 2', 'mirliton 3', 'off -No light- and chassis ID2 -2 feux verticaux-', 'pentagone blanc', 'pentagone inv', 'semaphore mecanique', 'triangle jaune pointe bas rappel 30', 'triangle jaune pointe haut ralentissement 30']
# desired_classes = ["fin de parcours", "pentagone blanc", "fleche jaune bleue", "jalon manoeuvre", "disque rouge mecanique", "coupez courant fin", "cercle jaune", "baissez panto distance", "demi cercle blanc", "baissez panto execution", "triangle jaune pointe haut ralentissement 30", "triangle jaune pointe bas rappel 30", "coupez courant", "fin de catenaire", "semaphore mecanique", "baissez panto fin", "drapeau jaune bleu", "carre violet", "demi cercle inv"]

# classes = get_class_index(desired_classes, names)

# copy_images_with_class(img_source_dir, label_source_dir, dest_dir, classes)
# print(classes)

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