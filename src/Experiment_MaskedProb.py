import torch
import torch.nn as nn
import os
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import evaluate
import pandas as pd
from transformers import ViTFeatureExtractor, AutoTokenizer, VisionEncoderDecoderModel

def matrix_to_tuple(matrix):
    """Converts the matrix to a tuple of tuples."""
    return tuple(tuple(row) for row in matrix)

def fill_matrix_and_store(matrix, row, col, combinations):
    """Recursively fills the matrix and stores the combinations in a set."""
    if row == 3:
        combinations.add(matrix_to_tuple(matrix))
        return

    if col == 3:
        fill_matrix_and_store(matrix, row + 1, 0, combinations)
        return

    # Set current cell to 0 and recurse
    matrix[row][col] = 0
    fill_matrix_and_store(matrix, row, col + 1, combinations)

    # Set current cell to 1 and recurse
    matrix[row][col] = 1
    fill_matrix_and_store(matrix, row, col + 1, combinations)

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

def compute_bleu(pred, gt):
    bleu = evaluate.load("google_bleu")
    pred_list = [pred]
    gt_list = [[gt]]

    bleu_score = bleu.compute(predictions=pred_list, references=gt_list)
    
    return bleu_score["google_bleu"]

t = VisionEncoderDecoderModel.from_pretrained('/opt/project/tmp/Image_Cationing_VIT_classification_v1.2')
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Initialize a 3x3 matrix with all zeros
initial_matrix = [[0 for _ in range(3)] for _ in range(3)]

# Set to store all unique combinations
combinations_set = set()

# Fill the matrix with all combinations and store them
fill_matrix_and_store(initial_matrix, 0, 0, combinations_set)

# Check the number of unique combinations
print(len(combinations_set)) 
# print(list(combinations_set)[:10])  # Display the size and first 5 combinations
print(list(combinations_set)[511][0])

patch_grid = 3
images_path = '/opt/project/dataset/Image/Testing/anomaly/'
experiment_caption = pd.read_excel('/opt/project/dataset/test_experiment_caption.xlsx', sheet_name='Sheet1')

for filename in os.listdir(images_path):
    collect = []
    if filename.endswith(".jpg"):
        # if file name in exeriment_caption == filename: extract the Caption in gt_caption
        gt_caption = experiment_caption.loc[experiment_caption['Filename'] == filename]['Caption'].values[0]
        print("Start image")
        original_image = cv2.imread(os.path.join(images_path, filename))
        image1 = Image.open(os.path.join(images_path, filename)).convert("RGB")
        
        original_width, original_height = image1.size
        patch_width = original_width / patch_grid
        patch_height = original_height / patch_grid

        for p in range(len(list(combinations_set))):
            masked_image = apply_masks_in_grid(original_image, list(combinations_set)[p], patch_grid, patch_width, patch_height)
            
            # Generate the caption for the image
            caption = tokenizer.decode(t.generate(feature_extractor(masked_image, return_tensors="pt").pixel_values)[0])

            # Remove [CLS] and [SEP] tokens from the caption
            tokens = caption.split()
            tokens_without_special_tokens = [token for token in tokens if token not in ["[CLS]", "[SEP]"]]
            caption_without_special_tokens = " ".join(tokens_without_special_tokens)
            
            bleu_score = compute_bleu(caption_without_special_tokens, gt_caption)*100
            
            # collect the p, bleu_score, and caption_without_special_tokens in list
            collect.append([p, bleu_score, caption_without_special_tokens])
        
        df = pd.DataFrame(collect)
        # save df to excel file and sheet name is image filename
        df.to_excel("/opt/project/tmp/result_maksed_experiment.xlsx", sheet_name=filename, index=False)
