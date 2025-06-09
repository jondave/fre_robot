# mediapipe_test_wsl.py (in your WSL Ubuntu)
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Use the device path directly, or 0 if it's the first video device
cap = cv2.VideoCapture(0) # Or cv2.VideoCapture('/dev/video0')

if not cap.isOpened():
    print("Error: Could not open webcam in WSL. Make sure it's attached via usbipd-win.")
    exit()

print("Webcam opened successfully in WSL.")

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

    print("Starting MediaPipe Pose detection loop. Press 'q' to quit.")
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break

        image = cv2.flip(image, 1) # Flip for selfie-view

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = pose.process(image_rgb)

        image_rgb.flags.writeable = True
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image_bgr,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        cv2.imshow('MediaPipe Pose Detection (WSL Direct)', image_bgr)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Script finished.")