import cv2

# Define mouse callback function
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Clicked at ({}, {})".format(x, y))

# Load image
image = cv2.imread("../data/roi_test1.jpg")

# Create window and set mouse callback function
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse_callback)

# Display image and wait for key press
cv2.imshow("Image", image)
cv2.waitKey(0)

# Destroy window
cv2.destroyAllWindows()