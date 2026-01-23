import cv2

# Create a video capture object. '0' usually refers to the default webcam.
# Change the index to 1, 2, etc., if you have multiple cameras.
cap = cv2.VideoCapture(0)

# Check if the camera was opened successfully.
if not cap.isOpened():
    print("Cannot open camera. Check if the index (0) is correct.")
    exit()

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # If the frame was not read correctly, break the loop.
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Display the frame in a window named 'Webcam Feed'
    cv2.imshow('Webcam Feed', frame)

    # Break the loop when the 'q' key is pressed.
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture object and destroy all OpenCV windows.
cap.release()
cv2.destroyAllWindows()
