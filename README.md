<div align="center">
  <h1> Synthetic Data for Object Detection </h1>
</div>

This repository contains a simple way to create french railway signs images and generate their label files in YOLO annotation format in order to train an object detection model. This could be used in any other field outside of railway signs.

Many of the railway signs templates used (including in this readme) are taken from the following repository: https://github.com/nicolaswurtz/signalisation-rfn-svg

## Overview:
The algorithm relies on python's albumnetations and OpenCv.

The idea is to take a background image, which is an already annotated image with only one sign, and put the new sign in the same position of the old sign based on its label file. 

**Examples of signs templates:** 
(Those are PNG vector images but one can use real life sign images cropped from their context, provided that the sign is clear enough.)

<div align="center">
  <img src="images/BP DIS.png" width=100>
  <img src="images/TIV PENDIS.png" width=100>
  <img src="images/ARRET A.png" width=100>
  <img src="images/Z.png" width=100>
  <img src="images/TIV D MOB.png" width=100>
</div>

**Examples of outputs:**
<div align="center">
  <img src="images/1.png" width=700>
  <img src="images/2.png" width=700>
  <img src="images/3.png" width=700>
  <img src="images/4.png" width=700>
</div>

## How to run:
To use the code, run the main.py file.

Modify the tranformations used in main.py depending on your specific need, and modify the size of the pasted sign in the SyntheticImage.py file.

**What you need: Files preparation**

- Choose backgrounds from real life data
- Fetch their corresponding labels files
- Prepare templates, PNG images of signs to be pasted on the background. Some of these samples were
taken from already annotated images, that were cropped and had their background removed, and some
other, especially classes that had little to no instances in real life images, were taken from computer drawn
SVG images
- Organise instances of each file in a separate folder named after the class label to allow automatic annotation

## Algorithm:
- Loop over folders to generate images for each sign one at time
- Select a random sample of the sign and a random background
- Generate a mask for the sign
- Retrieve, from the annotation file, the coordinates of the sign that is already in the selected background
and remove it using opencv’s inpaint function
- Apply different transformations to the sign object: Brightness contrast, RGBShift, rotation and Gaussian
Blur.
- Create the synthetic image by pasting the new sign where the previous sign was
- Apply some transformations to the final composition: Histogram Matching between the background and
the result to make the latter’s colors more natural looking, Gaussian Noise, Gaussian Blur, Perspective
Scale, Random Rain, Random shadow and Brightness contrast
- Create the label file of the composition based on the folder’s name, which is the class name, and the position
of the sign’s mask
