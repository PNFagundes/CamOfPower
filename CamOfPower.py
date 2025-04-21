import cv2
import numpy as np
import time

cap = cv2.VideoCapture(2) # Capture image from webcam

control_bg = False # Variable to control de background

last_time = time.time() # Last update
update_interval = 10  # Time interval

while True:
    _, frame = cap.read() # Read the image
    cropped_frame = frame[0:350, 0:640]
    
    if control_bg == False:
        frame_bg = np.copy(cropped_frame) # Create the initial background image
        control_bg = True
        
    frame_original = np.copy(cropped_frame) # Copy the original image

    hsv = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2HSV) # Convert to HSV
    
    # Define color ranges
    lower = np.array([0, 25, 70])
    upper = np.array([46, 154, 187])
    
    mask = cv2.inRange(hsv, lower, upper) # Create mask
    
    filter_imagem = cv2.bitwise_and(cropped_frame , cropped_frame , mask=mask) # Filter using the mask

    # Create the mask using morphological operations and Gaussian blur
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_dilated = cv2.dilate(mask_clean, kernel, iterations=4)
    mask_blur = cv2.GaussianBlur(mask_dilated, (5, 5), 0)
    mask_filled = cv2.morphologyEx(mask_blur, cv2.MORPH_CLOSE, kernel)
    
    
    filter_operations = cv2.bitwise_and(cropped_frame, filter_imagem, mask=mask_filled) # Filter using the processed mask
    
    # Create the new image
    new_image = np.copy(frame_original)
    mask = np.all(filter_operations != [0, 0, 0], axis=-1)
    new_image[mask] = frame_bg[mask]
    
    # Resetting the background
    current_time = time.time()
    if current_time - last_time >= update_interval and np.sum(filter_operations) <= 2000000:
        frame_bg = np.copy(frame_original)
        last_time = current_time
        cv2.putText(
            new_image,
            "Background updated",
            (50, 50),  # x and y positions
            cv2.FONT_HERSHEY_SIMPLEX,
            2,          # Font size
            (255, 0, 0),
            2,          # Line thickness
            cv2.LINE_AA
        )
        
    # Showing the images
    cv2.imshow("Original Image", frame_original)
    
    cv2.imshow("Filtered Image", filter_operations)
    
    cv2.imshow("New Imagem", new_image)
    
    # Exit the loop
    key = cv2.waitKey(1)
    if key == 27:
        break
    
cap.release() # Release the webcam resource
cv2.destroyAllWindows() # Close the windows