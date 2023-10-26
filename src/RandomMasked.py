from PIL import Image
import random

def remove_random_windows(image_path, window_size, num_windows_to_remove):
    # Open the image using Pillow
    img = Image.open(image_path)

    # Get image size
    width, height = img.size

    for _ in range(num_windows_to_remove):
        # Generate random positions for the top-left corner of the window
        x = random.randint(0, width - window_size)
        y = random.randint(0, height - window_size)

        # Create a rectangular mask with the window size
        mask = Image.new('L', (window_size, window_size), 255)

        # Paste the mask onto the original image at the specified position
        img.paste(mask, (x, y))

    # Save the modified image
    img.save('modified_image.jpg')
    img.show()

# for running module
if __name__ == "__main__":
    image_path = 'path_to_your_image.jpg'  # Replace with the actual path of your image
    window_size = 32
    num_windows_to_remove = 10
    remove_random_windows(image_path, window_size, num_windows_to_remove)
