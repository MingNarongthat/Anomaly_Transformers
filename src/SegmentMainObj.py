import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification
from torchvision import transforms, datasets
from PIL import Image, ImageDraw
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load the ViT model and feature extractor
model_name = "google/vit-base-patch16-224"
model = ViTForImageClassification.from_pretrained(model_name)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

# Ensure the model is in evaluation mode
model.eval()

# Define the data transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

# Load your custom dataset (without labels)
dataset_path = "/opt/project/dataset/ResNet50/Testing/landslide/"
dataset = [transform(Image.open(os.path.join(dataset_path, fname))) for fname in os.listdir(dataset_path) if fname.endswith(('.jpg', '.jpeg', '.png'))]
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

# Define a directory to save the segmented images
output_dir = "/opt/project/tmp/Segmented/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Counter for saved images
image_counter = 0


# Define a function to segment the image based on the model's confidence
def segment_image(image_tensor, model_confidence):
    # Check the number of channels and convert RGBA to RGB if necessary
    if image_tensor.shape[1] == 4:
        image_tensor = image_tensor[:, :3, :, :]  # Discard the alpha channel

    # Resize and normalize the image tensor
    resize_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to the range [-1, 1]
    ])
    processed_tensor = resize_transform(image_tensor[0])

    # Ensure the tensor has the shape (B, C, H, W)
    processed_tensor = processed_tensor.unsqueeze(0)

    with torch.no_grad():
        # Get model predictions
        outputs = model(processed_tensor)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        max_prob, predicted_label = probabilities.max(dim=-1)

        # Check if the model's confidence is above the threshold
        if max_prob.item() > model_confidence:
            # Convert tensor to PIL Image for drawing
            pil_image = transforms.ToPILImage()(image_tensor[0])

            # Draw a square around the object (for simplicity, we'll draw a square around the center of the image)
            draw = ImageDraw.Draw(pil_image)
            center_x, center_y = pil_image.size[0] // 2, pil_image.size[1] // 2
            half_side = 50  # half side length of the square
            draw.rectangle([(center_x - half_side, center_y - half_side), (center_x + half_side, center_y + half_side)],
                           outline="red", width=3)

            # Add text for significant level and type of object
            class_name = model.config.id2label[predicted_label.item()]
            draw.text((10, 10), f"Confidence: {max_prob.item():.2f}", fill="red")
            draw.text((10, 30), f"Class: {class_name}", fill="red")

            return pil_image
        else:
            return None

        # Iterate over your custom dataset
for image_tensor in dataloader:
    segmented_image = segment_image(image_tensor, model_confidence=0.6)
    if segmented_image:
        # Save the segmented image as JPG
        image_counter += 1
        file_name = f"segmented_{image_counter}.jpg"
        file_path = os.path.join(output_dir, file_name)
        segmented_image.save(file_path)
        print(f"Saved segmented image to {file_path}")
