import os
from PIL import Image
import cv2
import pandas as pd
import numpy as np

# Maskimg the image with the boxes
def apply_masks_in_grid(image, focus, patch_grid, patch_width, patch_height):
    # Make a copy of the image to keep the original intact
    masked_image = image.copy()
    
    for i in range(patch_grid):
            for j in range(patch_grid):
                xa, ya = (j * patch_width) + (patch_width / 2), (i * patch_height) + (patch_height / 2)
                wa, ha = patch_width, patch_height
        
                if focus[i][j] == 1:
                    # Apply the mask
                    cv2.rectangle(masked_image, (int(xa-wa/2), int(ya-ha/2)), (int(xa+wa/2), int(ya+ha/2)), (0, 0, 0), -1)
                else:
                    pass
        
    return masked_image

patch_grid = 3
images_path = '/opt/project/dataset/Image/Manual masked/'

for filename in os.listdir(images_path):
    if filename.endswith(".jpg"):
        # read the excel file
        data = pd.read_excel(os.path.join('/opt/project/tmp/', filename.replace('.jpg', '.xlsx')))
        # convert column 2 to float
        data[2] = data[2].astype(float)
        # sort the df[2] from highest to lowest
        df = data.sort_values(by=[2], ascending=False).head(4)
        print(filename, ':', np.mean(df[2]),df[3])

        #sort df[1] from lowest to highest 10 rows
        # df = data.sort_values(by=[1], ascending=True).head(10)
        
        # if file name in exeriment_caption == filename: extract 
        original_image = cv2.imread(os.path.join(images_path, filename))
        image1 = Image.open(os.path.join(images_path, filename)).convert("RGB")
        
        original_width, original_height = image1.size
        patch_width = original_width / patch_grid
        patch_height = original_height / patch_grid
        # loop excel file with the same image name
        for i in range(len(df)):
            masked_image = apply_masks_in_grid(original_image, eval(df.iloc[i][0]), patch_grid, patch_width, patch_height)
            cv2.imwrite('/opt/project/tmp/experiments/FigExperiment_{}_{}.jpg'.format(filename.replace('.jpg', ''), i), masked_image)
            
            
    
