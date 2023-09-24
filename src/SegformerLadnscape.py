import numpy as np
import cv2
from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation
from PIL import Image
import os

# Load model and feature extractor
feature_extractor = MaskFormerFeatureExtractor.from_pretrained("facebook/maskformer-swin-tiny-ade")
model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-tiny-ade")

# Load image from local dataset
images_path = "/opt/project/dataset/DataAll/Testing/"

for filename in os.listdir(images_path):
    if filename.endswith(".jpg"):
        image = Image.open(os.path.join(images_path, filename)).convert("RGB")

        # Convert PIL image to OpenCV format
        image_cv = np.array(image)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

        # Process image and get outputs
        inputs = feature_extractor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # Convert masks logits to probabilities
        mask_probs = outputs.masks_queries_logits.sigmoid().squeeze().cpu().detach().numpy()

        # Convert class logits to class labels
        _, labels = outputs.class_queries_logits.max(dim=-1)
        labels = labels.squeeze().cpu().numpy()

        # Iterate over each mask and draw rectangles on the original image
        for i, mask in enumerate(mask_probs):
            if labels[i] > 0:  # Ensure it's not the background class
                binary_mask = (mask > 0.995).astype(np.uint8)

                # Resize the binary mask to match the original image dimensions
                binary_mask_resized = cv2.resize(binary_mask, (image_cv.shape[1], image_cv.shape[0]),
                                                 interpolation=cv2.INTER_NEAREST)

                contours, _ = cv2.findContours(binary_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(image_cv, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Save the result with rectangles
        result_image = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
        result_image.save('/opt/project/tmp/semanticwithsquared{}.jpg'.format(filename))

