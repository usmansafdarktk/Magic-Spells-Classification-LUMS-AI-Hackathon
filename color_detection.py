import cv2

def get_hsv_range(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imshow("Select Ring Color", image)

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            hsv_value = hsv[y, x]
            print(f"HSV Value at ({x}, {y}): {hsv_value}")
            cv2.destroyAllWindows()

    cv2.setMouseCallback("Select Ring Color", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Provide the correct image path
get_hsv_range(r"C:\Users\muham\OneDrive\Pictures\Camera Roll\WIN_20241228_22_55_14_Pro.jpg")