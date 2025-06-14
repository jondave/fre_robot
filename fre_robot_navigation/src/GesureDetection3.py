import cv2
import mediapipe as mp
import numpy as np
import time
import math

# Initialize MediaPipe Pose solution
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def interpret_gesture(pose_landmarks) -> str:
    """
    Interprets human pose landmarks to determine a robot command.

    Args:
        pose_landmarks: The MediaPipe PoseLandmarks object containing detected keypoints.

    Returns:
        A string representing the robot command ('MOVE_FORWARD', 'REVERSE',
        'MOVE_LEFT', 'MOVE_RIGHT', 'STOP', 'NEUTRAL', or 'NO_COMMAND').
    """
    # Minimum visibility threshold for keypoints to be considered reliable.
    # Landmarks below this visibility will lead to a 'NO_COMMAND'.
    min_visibility = 0.7

    try:
        # Get relevant landmark positions. MediaPipe provides x, y, z, and visibility.
        # - y-coordinates typically increase downwards in image space (0.0 at top, 1.0 at bottom).
        # - x-coordinates typically increase rightwards in image space (0.0 at left, 1.0 at right).
        # All coordinates are normalized (0.0 to 1.0).
        left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        nose = pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]

        # Check visibility of critical landmarks to ensure confidence in pose detection.
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
            # Print which specific landmarks have low confidence for debugging.
            print(f"DEBUG: Low confidence in these landmarks: {', '.join(low_visibility_landmarks)}")
            return "NO_COMMAND" # Not enough confidence in pose to issue a command.

        # --- Define thresholds for gesture detection ---
        # UNCHANGED: Threshold for how high wrists need to be raised for the 'FORWARD' command.
        raise_arm_threshold = 0.1

        # MODIFIED: This factor now determines how far the wrist must be from the shoulder horizontally.
        # It no longer requires the arm to be vertically level with the shoulder.
        # A higher value means the arm must be extended further out.
        horizontal_arm_spread_factor = 0.1

        # UNCHANGED: Thresholds for the 'STOP' command's neutral pose remain the same for strictness.
        neutral_arm_y_threshold = 0.1
        neutral_arm_x_threshold = 0.1


        # --- Helper checks for arm positions relative to shoulders ---
        
        # MODIFIED: These checks are now much simpler. They only check if a wrist is extended
        # sideways, without requiring the arm to be perfectly horizontal. This makes the gesture
        # for turning and reversing much more natural.
        shoulder_width = abs(right_shoulder.x - left_shoulder.x)
        horizontal_extension_threshold = shoulder_width * horizontal_arm_spread_factor

        is_left_arm_extended_sideways = left_wrist.x < (left_shoulder.x - horizontal_extension_threshold)
        is_right_arm_extended_sideways = right_wrist.x > (right_shoulder.x + horizontal_extension_threshold)
        
        # UNCHANGED: Checks if an arm is in a generally "neutral" position for the STOP command.
        is_left_arm_neutral = (left_wrist.y > (left_shoulder.y - neutral_arm_y_threshold) and
                               abs(left_wrist.x - left_shoulder.x) < neutral_arm_x_threshold)
        is_right_arm_neutral = (right_wrist.y > (right_shoulder.y - neutral_arm_y_threshold) and
                                abs(right_wrist.x - right_shoulder.x) < neutral_arm_x_threshold)
      

        # --- Gesture Recognition Logic (Order of 'if/elif' statements dictates precedence) ---

        # Gesture 1: Both Arms Up -> Go Forward Straight (UNCHANGED)
        if (left_wrist.y < (left_shoulder.y - raise_arm_threshold) and
            right_wrist.y < (right_shoulder.y - raise_arm_threshold)):
            return "MOVE_FORWARD"

        # Gesture 2: T-shape (Both arms extended sideways) -> REVERSE (MODIFIED - LESS STRICT)
        # This check must come before the single-arm checks. It now uses the less strict "sideways" check.
        #elif is_left_arm_extended_sideways and is_right_arm_extended_sideways:
            return "REVERSE"
            
        # Gesture 3: Your physical LEFT arm extended sideways -> MOVE_LEFT (MODIFIED - LESS STRICT)
        # This no longer cares about the vertical position of the arm, only that it's extended out.
        #elif is_left_arm_extended_sideways:
            return "MOVE_LEFT"

        # Gesture 4: Your physical RIGHT arm extended sideways -> MOVE_RIGHT (MODIFIED - LESS STRICT)
        #elif is_right_arm_extended_sideways:
            return "MOVE_RIGHT"

        # Gesture 5: Neutral Pose (arms generally down and close to the body) -> STOP (UNCHANGED)
        elif is_left_arm_neutral and is_right_arm_neutral:
            return "STOP"
        
        elif is_left_arm_neutral and right_wrist.y < (right_shoulder.y - raise_arm_threshold):
            return "TURN RIGHT"

        elif is_right_arm_neutral and left_wrist.y < (left_shoulder.y - raise_arm_threshold):
            return "TURN LEFT"

        # If no specific gesture matches any of the above conditions, return "NO_COMMAND".
        return "NO_COMMAND"

    except Exception as e:
        # Catch any errors during landmark processing (e.g., if landmark objects are None unexpectedly).
        print(f"Error processing pose landmarks: {e}")
        return "NO_COMMAND"


# --- Main Script Execution ---

# Initialize video capture. '0' usually refers to the default webcam.
# You might need to try '1', '-1', or a specific device path (e.g., '/dev/video0' on Linux/WSL)
# if your webcam isn't found at index 0.
cap = cv2.VideoCapture(0)

# Check if the webcam opened successfully.
if not cap.isOpened():
    print("Error: Could not open webcam. Make sure it's not in use and try different camera indices (0, 1, -1).")
    print("If using WSL, ensure your X server is running and DISPLAY variable is set correctly.")
    exit() # Exit the script if the camera cannot be opened.

print("Webcam opened successfully.")

# Variables for debouncing commands:
# `last_detected_command`: Stores the last command that was detected (even if not yet printed/sent).
# `command_start_time`: Timestamp when the `last_detected_command` was first detected.
# `command_debounce_delay`: The duration (in seconds) a command must be held continuously
# before it's registered as a stable command. This prevents flickering.
last_detected_command = ""
command_start_time = 0
command_debounce_delay = 0.5 # A command must be held for 0.5 seconds to be registered.

# Initialize MediaPipe Pose processing.
# `min_detection_confidence`: Minimum confidence for a person detection to be considered valid (0.0 to 1.0).
# `min_tracking_confidence`: Minimum confidence for a tracked pose landmark to be considered valid.
# (Lowering these can help with detection in challenging conditions but might increase false positives).
with mp_pose.Pose(
    min_detection_confidence=0.3, # User-specified confidence level
    min_tracking_confidence=0.3) as pose:

    print("\nStarting MediaPipe Pose detection loop. Press 'q' to quit.")
    print("--- Robot Commands (Printed to Terminal) ---") # Header for command output.

    # Main loop for video processing.
    while cap.isOpened():
        # Read a frame from the webcam.
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame or camera disconnected. Retrying...")
            time.sleep(0.1) # Small delay before trying to read again.
            continue # Skip to the next iteration of the loop.

        # Flip the image horizontally (mirror effect) for a more intuitive "selfie-view" display.
        # This means your left arm appears on the right side of the screen, and vice-versa.
        image = cv2.flip(image, 1)

        # Convert the image from BGR (OpenCV default) to RGB (MediaPipe required).
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Set image as not writeable to improve performance (MediaPipe processes in-place).
        image_rgb.flags.writeable = False
        # Process the image with MediaPipe Pose to detect landmarks.
        results = pose.process(image_rgb)

        # Re-enable writeability for drawing.
        image_rgb.flags.writeable = True
        # Convert the image back to BGR for OpenCV display.
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        current_raw_command = "NO_COMMAND" # Default command if no pose or gesture is detected.
        if results.pose_landmarks:
            # Draw the detected pose landmarks on the image for visualization.
            mp_drawing.draw_landmarks(
                image_bgr,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            # Interpret the detected pose into a robot command string.
            current_raw_command = interpret_gesture(results.pose_landmarks)
        
        # --- Command Debouncing Logic ---
        # This ensures a command is only printed/sent if it's held consistently for a duration.
        if current_raw_command != last_detected_command:
            # If the detected command changes, reset the timer.
            last_detected_command = current_raw_command
            command_start_time = time.time()
        
        # If the current command has been stable (held for longer than the debounce delay).
        if time.time() - command_start_time > command_debounce_delay:
            # Print the stable command to the terminal.
            # (In a real robot application, this is where you'd send the command to your robot).
            print(f"DEBUG_COMMAND: {current_raw_command}")
            # Optionally, you might reset command_start_time here if you want the command to print
            # repeatedly every `debounce_delay` seconds while held, rather than just once after stabilization.
            # command_start_time = time.time() 
        
        # Display the video feed with landmarks and annotations.
        cv2.imshow('MediaPipe Pose Detection', image_bgr)

        # Wait for 5 milliseconds for a key press. If 'q' is pressed, break the loop to quit.
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Release the webcam resource.
cap.release()
# Close all OpenCV display windows.
cv2.destroyAllWindows()
print("\nScript finished. Webcam and OpenCV windows closed.")
