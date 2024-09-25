import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import albumentations as A
import time
import random
from tqdm import tqdm
from utils_ import * 
import string 

class SyntheticImage:

    def __init__(self, background_path, sign_path):
        """
        Constructor method for the class Synthetic Image

        Parameters:
        -----------
        background_path : str
            Path for the image on which we want to put the sign
        
        sign_path : str
            Path for the railway sign that we want to put on the background


        Attributes:
        -----------
        image: numpy array
            Result of merging the sign and the background
        
        mask: numpy array
            Mask of the sign 
        
        background_path: str

        background: numpy array
            background on which we want to paste the sign

        sign_path: str

        sign: numpy array

        label: str
            Class of the sign retrieved from its original folder

        sign_coordinates: list
            annotation in yolo format of the image

        """

        self.image = None
        self.mask = None
        self.background_path = background_path
        self.background = cv2.imread(self.background_path)
        self.sign_path = sign_path
        self.sign = cv2.imread(self.sign_path, cv2.IMREAD_UNCHANGED)
        self.label = None
        self.sign_coordinate = None


    def get_img_and_mask(self):
        """
        Reads the sign image in rgba format and make a corresponding mask for the sign
        """
        # Extract RGB and alpha, convert to float, normalize alpha, 
        # apply alpha to RGB
        mask = self.sign[:, :, 3]
        self.mask = np.where(mask>0, 1, 0).astype(np.uint8)
        # self.sign = cv2.multiply(self.mask, self.sign[:, :, :3])
        self.sign = cv2.bitwise_and(self.sign[:, :, :3], self.sign[:, :, :3], self.mask)
        

    def resize_transform_obj(self, new_h, transforms):
        """
        Resizes the image while keeping the original height/width ratio 
        and transforms using Albumentation library transforms
        Changes are applied to both the mask and the image
        """
        # random_ex = random.randint(1, 20)
        #### TO MODIFY the size max_size=int(new_h*1.5) depending on your use case
        transform_resize =   transform_resize = A.LongestMaxSize(max_size=int(new_h*1.5), interpolation=1, always_apply=True)
        transformed_resized = transform_resize(image=self.sign, mask=self.mask)
        img_t = transformed_resized["image"]
        mask_t = transformed_resized["mask"]
            
        transformed = transforms(image=img_t, mask=mask_t)
        img_t = transformed["image"]
        mask_t = transformed["mask"]
        
        self.sign = img_t
        self.mask = mask_t

    def transform_background(self, bbox_coor):
        """
        Tries to erase the old sign in the background image using cv2's inpaint
        """
        mask = np.zeros(self.background.shape[:2], np.uint8)
        xmin, ymin, xmax, ymax = bbox_coor
        mask[int(ymin):int(ymax), int(xmin):int(xmax)] = 255 
        self.background =  cv2.inpaint(self.background, mask, 3, cv2.INPAINT_NS)
    
    def add_obj(self, mask_comp, mask, x, y):
        '''
        img_comp - composition of objects
        mask_comp - composition of objects` masks
        img - image of object
        mask - binary mask of object
        x, y - coordinates where center of img is placed
        Function returns img_comp in CV2 RGB format + mask_comp
        '''
        img_comp = self.background.copy()
        img = self.sign
        h_comp, w_comp = img_comp.shape[0], img_comp.shape[1]
        h, w = self.sign.shape[0], self.sign.shape[1]
        
        x = x - int(w/2)
        y = y - int(h/2)
        
        mask_b = mask == 1
        mask_rgb_b = np.stack([mask_b, mask_b, mask_b], axis=2)

        if x >= 0 and y >= 0:
            h_part = h - max(0, y+h-h_comp) # h_part - part of the image which gets into the frame of img_comp along y-axis
            w_part = w - max(0, x+w-w_comp) # w_part - part of the image which gets into the frame of img_comp along x-axis

            img_comp[y:y+h_part, x:x+w_part, :] = img_comp[y:y+h_part, x:x+w_part, :] * ~mask_rgb_b[0:h_part, 0:w_part, :] + (img * mask_rgb_b)[0:h_part, 0:w_part, :]
            mask_comp[y:y+h_part, x:x+w_part] = mask_comp[y:y+h_part, x:x+w_part] * ~mask_b[0:h_part, 0:w_part] + (mask_b)[0:h_part, 0:w_part]
            mask_added = mask[0:h_part, 0:w_part]
            
        elif x < 0 and y < 0:
            h_part = h + y
            w_part = w + x
            
            img_comp[0:0+h_part, 0:0+w_part, :] = img_comp[0:0+h_part, 0:0+w_part, :] * ~mask_rgb_b[h-h_part:h, w-w_part:w, :] + (img * mask_rgb_b)[h-h_part:h, w-w_part:w, :]
            mask_comp[0:0+h_part, 0:0+w_part] = mask_comp[0:0+h_part, 0:0+w_part] * ~mask_b[h-h_part:h, w-w_part:w] + (mask_b)[h-h_part:h, w-w_part:w]
            mask_added = mask[h-h_part:h, w-w_part:w]
            
        elif x < 0 and y >= 0:    
            h_part = h - max(0, y+h-h_comp)
            w_part = w + x
            
            img_comp[y:y+h_part, 0:0+w_part, :] = img_comp[y:y+h_part, 0:0+w_part, :] * ~mask_rgb_b[0:h_part, w-w_part:w, :] + (img * mask_rgb_b)[0:h_part, w-w_part:w, :]
            mask_comp[y:y+h_part, 0:0+w_part] = mask_comp[y:y+h_part, 0:0+w_part] * ~mask_b[0:h_part, w-w_part:w] + (mask_b)[0:h_part, w-w_part:w]
            mask_added = mask[0:h_part, w-w_part:w]
            
        elif x >= 0 and y < 0: 
            h_part = h + y
            w_part = w - max(0, x+w-w_comp)
            
            img_comp[0:0+h_part, x:x+w_part, :] = img_comp[0:0+h_part, x:x+w_part, :] * ~mask_rgb_b[h-h_part:h, 0:w_part, :] + (img * mask_rgb_b)[h-h_part:h, 0:w_part, :]
            mask_comp[0:0+h_part, x:x+w_part] = mask_comp[0:0+h_part, x:x+w_part] * ~mask_b[h-h_part:h, 0:w_part] + (mask_b)[h-h_part:h, 0:w_part]
            mask_added = mask[h-h_part:h, 0:w_part]
        
        return img_comp, mask_comp, mask_added

    def retrieve_yolo_coordinates(self, labels_dir, labels):
        """
        Get yolo coordinates of the old sign in the background image
        """
        image_name, _ = os.path.splitext(os.path.basename(self.background_path))
        image_name = image_name.replace(" ", "-")
        image_name = image_name.replace("’", "-")
        image_name = image_name.replace("é", "e")

        for file in labels:
            if file.startswith(image_name):
                with open(labels_dir + "/" + file, 'r') as file:
                    line = file.readline()
                    return line.split()

    def yolo_to_bbox(self, yolo_coordinates):
        """
        get xmin, ymin, xmax, ymax coordinates for the new sign based on the bounding box of the previous sign.
        Result is retrived from the annotation's bouding boxes
        """

        _, x_center, y_center, width, height = yolo_coordinates
        h, w, _ = self.background.shape

        # # Calculate half width and half height
        # half_width = float(width) / 2
        # half_height = float(height) / 2

        # # Convert center coordinates to top-left corner
        # xmin = float(x_center) - half_width
        # ymin = float(y_center) - half_height
        
        # # Normalize to pixel values
        # xmin *= image_width
        # ymin *= image_height
        # half_width *= image_width
        # half_height *= image_height

        # # Calculate bottom-right corner coordinates
        # xmax = xmin + float(width) * image_width
        # ymax = ymin + float(height) * image_height

        return w*float(x_center), h*float(y_center), w*float(width), h*float(height)

    def yolo_to_bbox_(self, yolo_coordinates):
        _, x_center, y_center, width, height = yolo_coordinates
        h, w, _ = self.background.shape

        # Calculate half width and half height
        half_width = float(width) / 2
        half_height = float(height) / 2

        # Convert center coordinates to top-left corner
        xmin = float(x_center) - half_width
        ymin = float(y_center) - half_height
        
        # Normalize to pixel values
        xmin *= w
        ymin *= h
        half_width *= w
        half_height *= h

        # Calculate bottom-right corner coordinates
        xmax = xmin + float(width) * w
        ymax = ymin + float(height) * h

        return xmin, ymin, xmax, ymax

    def create_composition(self, x_center, y_center, transform):
        h, w = self.background.shape[0], self.background.shape[1]
        mask_comp = np.zeros((h,w), dtype=np.uint8)
        img_comp, mask_comp, _ = self.add_obj(mask_comp, self.mask, x=x_center, y=y_center)     
    
        self.image, self.mask = transform(image=img_comp)["image"], mask_comp

    
    def generate_random_name(self, length=10):
        """Generates a random string of letters and digits of specified length."""
        letters_and_digits = string.ascii_letters + string.digits
        return ''.join(random.choice(letters_and_digits) for _ in range(length))

    
    def create_yolo_annotations(self, folder_name):
        # mask_w, mask_h = self.mask.shape[1], self.mask.shape[0]
        image_w, image_h = self.image.shape[1], self.image.shape[0]
        
        # obj_ids = np.unique(self.mask).astype(np.uint8)[1:]
        # mask = self.mask == obj_ids[:, None, None]

        pos = np.where(self.mask)
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])

        xc = (xmin + xmax) / 2
        yc = (ymin + ymax) / 2
        w = xmax - xmin
        h = ymax - ymin

        annotations_yolo = [folder_name,
                                round(xc/image_w, 5),
                                round(yc/image_h, 5),
                                round(w/image_w, 5),
                                round(h/image_h, 5)]
        
        return annotations_yolo
    
    
    def save_composition_and_annotation(self, output_path, folder_name):
        name = self.generate_random_name()
        cv2.imwrite(f"{output_path}/images/{name}.png", self.image)
        
        annotations = self.create_yolo_annotations(folder_name)
        with open(f"{output_path}/labels/{name}.txt", "a") as f:
                f.write(' '.join(str(el) for el in annotations) + '\n')