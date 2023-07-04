import cv2

# Load the pre-trained YOLO model
net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')

# Load the class labels
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Load the image
image = cv2.imread('/opt/project/dataset/Image/Testing/anomaly/frameHf-LtoYEPDw_000000_000015.jpg')

# Create a blob from the image
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

# Set the blob as input to the network
net.setInput(blob)

# Run forward pass and get the output layer names
output_layers = net.getUnconnectedOutLayersNames()
layer_outputs = net.forward(output_layers)

# Initialize lists for bounding boxes, confidences, and class IDs
boxes = []
confidences = []
class_ids = []

# Loop over each output layer
for output in layer_outputs:
    # Loop over each detection
    for detection in output:
        # Extract class ID and confidence
        scores = detection[5:]
        class_id = scores.argmax()
        confidence = scores[class_id]

        # Filter out weak detections
        if confidence > 0.5:
            # Scale the bounding box coordinates
            width, height = image.shape[1], image.shape[0]
            center_x, center_y, bbox_width, bbox_height = detection[:4] * [width, height, width, height]
            x, y = int(center_x - bbox_width / 2), int(center_y - bbox_height / 2)

            # Add bounding box, confidence, and class ID to respective lists
            boxes.append([x, y, int(bbox_width), int(bbox_height)])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply non-maximum suppression to remove overlapping boxes
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Draw bounding boxes and labels on the image
for i in indices:
    i = i[0]
    x, y, w, h = boxes[i]
    label = classes[class_ids[i]]
    confidence = confidences[i]

    # Draw the bounding box and label
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, f"{label}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save the output image with bounding boxes
cv2.imwrite('/opt/project/tmp/output.jpg', image)
