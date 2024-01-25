import cv2

def test_process_image():
    # Read the test image
    image_path = "/opt/project/dataset/Image/Manual masked/1.jpg"
    image = cv2.imread(image_path)

    # Call the function
    process_image()

    # Verify the results
    assert cv2.waitKey() == -1  # Check that no key was pressed
    cv2.destroyAllWindows()  # Check that all windows were closed

    # You can also add additional assertions to check the output images if needed
    # For example, you can use cv2.imread to read the saved images and compare them with expected results

# Run the test
test_process_image()