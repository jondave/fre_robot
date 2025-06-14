import cv2
import mediapipe as mp
import numpy as np
import time

# ROS2 imports
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image  # Import the Image message type
from cv_bridge import CvBridge, CvBridgeError  # Import CvBridge and CvBridgeError

# Initialize MediaPipe Pose solution
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

#------------------------------------------------------------------------------------------------------------------

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
            # self.get_logger().info(f"DEBUG: Low confidence in these landmarks: {', '.join(low_visibility_landmarks)}") # Use get_logger()
            return "NO_COMMAND" # Not enough confidence in pose to issue a command.

        # --- Define thresholds for gesture detection ---
        raise_arm_threshold = 0.1
        horizontal_arm_spread_factor = 0.1
        neutral_arm_y_threshold = 0.1
        neutral_arm_x_threshold = 0.1

        # --- Helper checks for arm positions relative to shoulders ---
        shoulder_width = abs(right_shoulder.x - left_shoulder.x)
        horizontal_extension_threshold = shoulder_width * horizontal_arm_spread_factor

        is_left_arm_extended_sideways = left_wrist.x < (left_shoulder.x - horizontal_extension_threshold)
        is_right_arm_extended_sideways = right_wrist.x > (right_shoulder.x + horizontal_extension_threshold)
        
        is_left_arm_neutral = (left_wrist.y > (left_shoulder.y - neutral_arm_y_threshold) and
                                 abs(left_wrist.x - left_shoulder.x) < neutral_arm_x_threshold)
        is_right_arm_neutral = (right_wrist.y > (right_shoulder.y - neutral_arm_y_threshold) and
                                 abs(right_wrist.x - right_shoulder.x) < neutral_arm_x_threshold)
      

        # --- Gesture Recognition Logic (Order of 'if/elif' statements dictates precedence) ---

        if (left_wrist.y < (left_shoulder.y - raise_arm_threshold) and
            right_wrist.y < (right_shoulder.y - raise_arm_threshold)):
            return "REVERSE"

        elif is_left_arm_neutral and is_right_arm_neutral:
            return "STOP"
        
        elif is_left_arm_neutral and right_wrist.y < (right_shoulder.y - raise_arm_threshold):
            return "TURN RIGHT"

        elif is_right_arm_neutral and left_wrist.y < (left_shoulder.y - raise_arm_threshold):
            return "TURN LEFT"

        return "NO_COMMAND"

    except Exception as e:
        # self.get_logger().error(f"Error processing pose landmarks: {e}") # Use get_logger()
        return "NO_COMMAND"

class GesturePublisher(Node):
    def __init__(self):
        super().__init__('gesture_publisher')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # New publisher for the processed image
        self.image_publisher = self.create_publisher(Image, '/mediapipe_pose_image', 10)

        self.last_detected_command = ""
        self.command_start_time = 0
        self.command_debounce_delay = 0.5 # seconds

        # Create a subscriber for the RealSense camera image topic
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',  # Common RealSense color image topic. Adjust if yours is different.
            self.image_callback,
            10)
        self.subscription  # prevent unused variable warning

        # Initialize CvBridge
        self.br = CvBridge()

        # Initialize MediaPipe Pose
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3)
        
        self.get_logger().info("ROS2 Gesture Publisher node initialized.")
        self.get_logger().info("Subscribing to /camera/camera/color/image_raw for input images.")
        self.get_logger().info("Publishing processed images to /mediapipe_pose_image.")
        self.get_logger().info("--- Robot Commands (Printed to Terminal) ---")

    def publish_twist(self, command):
        twist_msg = Twist()
        if command == "REVERSE":
            twist_msg.linear.x = -0.2
            twist_msg.angular.z = 0.0
            self.get_logger().info("Command: robot is REVERSE")
        elif command == "TURN LEFT":
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = np.pi / 2
            self.get_logger().info("Command: robot is TURNING LEFT")
        elif command == "TURN RIGHT":
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = -np.pi / 2
            self.get_logger().info("Command: robot is TURNING RIGHT")
        elif command == "STOP":
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = 0.0
            self.get_logger().info("Command: robot is STOPPED")
        else:
            twist_msg.linear.x = 0.2
            twist_msg.angular.z = 0.0
            self.get_logger().info("Command: robot is MOVING FORWARD")

        self.publisher.publish(twist_msg)

    def image_callback(self, data):
        """
        Callback function for the image subscription.
        Converts the ROS Image message to an OpenCV image, processes it,
        and publishes the result to a new topic.
        """
        try:
            # Convert ROS Image message to OpenCV image (bgr8 encoding for color)
            current_frame = self.br.imgmsg_to_cv2(data, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return

        # Flip the image horizontally for selfie-view display (optional, can be removed)
        image = cv2.flip(current_frame, 1)

        # Convert the BGR image to RGB before processing with MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False # Optimize performance
        results = self.pose.process(image_rgb)
        image_rgb.flags.writeable = True
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR) # Convert back to BGR for drawing/publishing

        current_raw_command = "NO_COMMAND"
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image_bgr,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            current_raw_command = interpret_gesture(results.pose_landmarks)

        # Debouncing logic for robot commands
        if current_raw_command != self.last_detected_command:
            self.last_detected_command = current_raw_command
            self.command_start_time = time.time()

        if time.time() - self.command_start_time > self.command_debounce_delay:
            # self.get_logger().info(f"DEBUG_COMMAND: {current_raw_command}")
            self.publish_twist(current_raw_command)
            # self.command_start_time = time.time() # Uncomment to publish command continuously

        # Publish the processed image
        try:
            self.image_publisher.publish(self.br.cv2_to_imgmsg(image_bgr, "bgr8"))
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")


def main(args=None):
    rclpy.init(args=args)
    gesture_publisher = GesturePublisher()

    try:
        rclpy.spin(gesture_publisher)
    except KeyboardInterrupt:
        gesture_publisher.get_logger().info('Shutting down Gesture Publisher node...')
    finally:
        gesture_publisher.pose.close() # Ensure MediaPipe Pose solution is properly closed
        # No need for cv2.destroyAllWindows() as no windows are shown
        rclpy.shutdown()

if __name__ == '__main__':
    main()