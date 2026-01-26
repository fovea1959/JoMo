import cv2

import CameraFinder

# Initialize webcam
cam_index, cam_backend = camera_finder.get_camera('Integrated')
cap = cv2.VideoCapture(cam_index, cam_backend)

# Initialize background model
background_model = None
alpha = 0.02  # Learning rate for running average

while True:
    ret, frame = cap.read()
    if not ret: break

    cv2.imshow("Frame", frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)  # Reduce noise

    # Initialize running average
    if background_model is None:
        background_model = gray.copy().astype("float")
        continue

    # Update running average: weighted sum
    cv2.accumulateWeighted(gray, background_model, alpha)
    # Convert back to uint8 for difference calculation
    frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(background_model))

    # Threshold the image
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    cv2.imshow("Motion", thresh)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
