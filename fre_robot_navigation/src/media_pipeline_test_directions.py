import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Pose solution
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def interpret_gesture(pose_landmarks) -> str:
    min_visibility = 0.7

    try:
        left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        nose = pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]

        # --- DEBUG PRINTS START ---
        # Uncomment these lines if you want to see the coordinate data
        # print(f"DEBUG: Left Shoulder (x,y): ({left_shoulder.x:.2f}, {left_shoulder.y:.2f})")
        # print(f"DEBUG: Right Shoulder (x,y): ({right_shoulder.x:.2f}, {right_shoulder.y:.2f})")
        # print(f"DEBUG: Left Wrist (x,y): ({left_wrist.x:.2f}, {left_wrist.y:.2f})")
        # print(f"DEBUG: Right Wrist (x,y): ({right_wrist.x:.2f}, {right_wrist.y:.2f})")
        # print(f"DEBUG: Nose (x,y): ({nose.x:.2f}, {nose.y:.2f})")
        # --- DEBUG PRINTS END ---

        # Check visibility of critical landmarks
        low_visibility_landmarks = []
        critical_landmarks_map = {
            "Left Shoulder": left_shoulder, "Right Shoulder": right_shoulder,
            "Left Wrist": left_wrist, "Right Wrist": right_wrist,
            "Left Elbow": left_elbow, "Right Elbow": right_elbow,
            "Nose": nose
        }

        for name, landmark in critical_landmarks_map.items():
            if landmark.visibility < min_visibility:
                low_visibility_landmarks.append(f"{name} (vis: {landmark.visibility:.2f})")

        if low_visibility_landmarks:
            print(f"DEBUG: Low confidence in these landmarks: {', '.join(low_visibility_landmarks)}")
            return "NO_COMMAND"

        # Define thresholds for gesture detection
        vertical_threshold = 0.1 # Max difference in Y to consider arm 'straight' horizontally
        raise_arm_threshold = 0.15 # How much wrist Y should be above shoulder Y to be 'up' (for MOVE_FORWARD)
        horizontal_arm_spread_factor = 0.2 # How far out wrists should be from shoulders for T-pose/single arm outstretched

        neutral_arm_y_threshold = 0.1 # How much wrist Y can be below shoulder Y for neutral
        neutral_arm_x_threshold = 0.1 # How much wrist X can be away from shoulder X for neutral

        # Helper checks for horizontal arm position
        is_left_arm_horizontal = (
            abs(left_wrist.y - left_shoulder.y) < vertical_threshold and
            abs(left_elbow.y - left_shoulder.y) < vertical_threshold and
            left_wrist.x < (left_shoulder.x - (right_shoulder.x - left_shoulder.x) * horizontal_arm_spread_factor) # Outstretched left
        )
        is_right_arm_horizontal = (
            abs(right_wrist.y - right_shoulder.y) < vertical_threshold and
            abs(right_elbow.y - right_shoulder.y) < vertical_threshold and
            right_wrist.x > (right_shoulder.x + (right_shoulder.x - left_shoulder.x) * horizontal_arm_spread_factor) # Outstretched right
        )
        
        # Helper check for arm being down/neutral
        is_left_arm_down = left_wrist.y > left_shoulder.y + neutral_arm_y_threshold
        is_right_arm_down = right_wrist.y > right_shoulder.y + neutral_arm_y_threshold


        # Gesture 1: Both Arms Up -> Go Forward Straight
        if (left_wrist.y < (left_shoulder.y - raise_arm_threshold) and
            right_wrist.y < (right_shoulder.y - raise_arm_threshold)):
            return "MOVE_FORWARD"

        # Gesture 2: Right Arm Horizontal (on screen, your left arm) -> Move Left
        # Requires the left arm (MediaPipe's LEFT_WRIST) to be horizontal and the right arm to be down
        elif is_left_arm_horizontal and is_right_arm_down:
            return "MOVE_LEFT"

        # Gesture 3: Left Arm Horizontal (on screen, your right arm) -> Move Right
        # Requires the right arm (MediaPipe's RIGHT_WRIST) to be horizontal and the left arm to be down
        elif is_right_arm_horizontal and is_left_arm_down:
            return "MOVE_RIGHT"

        # Gesture 4: T-shape (Both arms roughly horizontal and outstretched) -> Stop
        # This condition already uses the 'horizontal_arm_spread_factor'
        # We need to make sure this doesn't conflict with single arm horizontal.
        # The current `arms_outstretched` checks both left_wrist.x and right_wrist.x
        # and `arms_horizontal_left` and `arms_horizontal_right` are also checking both.
        # So, the original logic for STOP (requiring BOTH arms to be horizontal and outstretched) is fine.
        arms_horizontal_left_stop = abs(left_wrist.y - left_shoulder.y) < vertical_threshold and \
                               abs(left_elbow.y - left_shoulder.y) < vertical_threshold
        arms_horizontal_right_stop = abs(right_wrist.y - right_shoulder.y) < vertical_threshold and \
                                abs(right_elbow.y - right_shoulder.y) < vertical_threshold

        arms_outstretched_stop = left_wrist.x < (left_shoulder.x - (right_shoulder.x - left_shoulder.x) * horizontal_arm_spread_factor) and \
                            right_wrist.x > (right_shoulder.x + (right_shoulder.x - left_shoulder.x) * horizontal_arm_spread_factor)

        if arms_horizontal_left_stop and arms_horizontal_right_stop and arms_outstretched_stop:
            return "STOP"
            
        # Gesture 5: Neutral Pose
        is_left_arm_neutral = (left_wrist.y > (left_shoulder.y - neutral_arm_y_threshold) and
                               abs(left_wrist.x - left_shoulder.x) < neutral_arm_x_threshold)
        is_right_arm_neutral = (right_wrist.y > (right_shoulder.y - neutral_arm_y_threshold) and
                                abs(right_wrist.x - right_shoulder.x) < neutral_arm_x_threshold)

        if is_left_arm_neutral and is_right_arm_neutral:
            return "NEUTRAL"

        return "NO_COMMAND"

    except Exception as e:
        print(f"Error processing pose landmarks: {e}")
        return "NO_COMMAND"


# --- Main Script (no changes needed here, assuming you're using the latest version) ---

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam. Make sure it's not in use and try different camera indices (0, 1, -1).")
    exit()

print("Webcam opened successfully.")

last_detected_command = ""
command_start_time = 0
command_debounce_delay = 0.5 # Seconds a command must be held to be registered

with mp_pose.Pose(
    min_detection_confidence=0.3, # Using your updated confidence levels
    min_tracking_confidence=0.3) as pose:

    print("\nStarting MediaPipe Pose detection loop. Press 'q' to quit.")
    print("HELLO")
    print("--- Robot Commands (Printed to Terminal) ---")
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame or camera disconnected.")
            time.sleep(0.1)
            continue

        # image = cv2.flip(image, 1) # Keep this for mirror view

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = pose.process(image_rgb)

        image_rgb.flags.writeable = True
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        current_raw_command = "NO_COMMAND"
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image_bgr,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            current_raw_command = interpret_gesture(results.pose_landmarks)
        
        # Debouncing logic
        if current_raw_command != last_detected_command:
            last_detected_command = current_raw_command
            command_start_time = time.time()
        
        if time.time() - command_start_time > command_debounce_delay:
            print(f"DEBUG_COMMAND: {current_raw_command}")
        
        cv2.imshow('MediaPipe Pose Detection', image_bgr)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("\nScript finished. Webcam and OpenCV windows closed.")