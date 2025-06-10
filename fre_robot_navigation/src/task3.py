import rclpy
from rclpy.node import Node
from math import sin, cos, pi
from geometry_msgs.msg import PoseStamped, Quaternion, PoseWithCovarianceStamped
from sensor_msgs.msg import Joy
from tf_transformations import quaternion_from_euler
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.action.client import ClientGoalHandle

# to run this node, use:
# ros2 launch fre_robot_bringup zero_teleop.launch.py

# ros2 launch fre_robot_bringup accessories.launch.py
# ros2 launch  fre_robot_bringup realsense_1.launch.py

# ros2 launch fre_robot_navigation ekf_localisation_local.launch.py

# ros2 launch roverrobotics_driver slam_launch.py
# ros2 launch nav2_bringup navigation_launch.py

# TODO add fruit_detector_v2.py

class TreeNavigator(Node):
    def __init__(self):
        super().__init__('tree_navigator')

        # Publisher for initial pose
        self.initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            '/initialpose',
            QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST)
        )
        self.publish_initial_pose(1.0, 1.0, 0.0)

        # Subscribe to joystick topic for pause/resume buttons
        self.joy_sub = self.create_subscription(
            Joy,
            '/joy',
            self.joy_callback,
            10
        )

        # Create action client for Nav2 NavigateToPose
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.get_logger().info('Waiting for nav2 action server...')
        self.nav_to_pose_client.wait_for_server()
        self.get_logger().info('Nav2 Action server available.')

        # Define tree positions
        self.tree_positions = [
            {'fruit': 'apple', 'x': 2.0, 'y': 2.0},
            {'fruit': 'lemon', 'x': 5.0, 'y': 1.5},
            {'fruit': 'banana', 'x': 8.0, 'y': 3.0},
            {'fruit': 'grapes', 'x': 3.0, 'y': 7.0},
            {'fruit': 'orange', 'x': 7.5, 'y': 8.0}
        ]

        self.circle_radius = 0.75
        self.num_points = 8

        # State management for mission
        self.paused = True  # Start in paused state
        self.mission_started = False # Mission has not started yet
        self.current_tree_idx = 0
        self.current_point_idx = 0
        self.current_goal_handle: ClientGoalHandle = None
        self.goal_active = False # Flag to track if a goal is actively being pursued

        # Timer to run mission step-by-step
        self.timer = self.create_timer(0.5, self.execute_mission_step)
        self.get_logger().info('Mission is paused. Press resume button to start.')


    def publish_initial_pose(self, x, y, yaw):
        initial_pose = PoseWithCovarianceStamped()
        initial_pose.header.frame_id = 'map'
        initial_pose.header.stamp = self.get_clock().now().to_msg()
        # FIX: Corrected access to position and orientation
        initial_pose.pose.pose.position.x = x
        initial_pose.pose.pose.position.y = y
        initial_pose.pose.pose.position.z = 0.0

        quat = quaternion_from_euler(0, 0, yaw)
        initial_pose.pose.pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
        initial_pose.pose.covariance = [
            0.25, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.25, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0685, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0685, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0685, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0685
        ]
        self.initial_pose_pub.publish(initial_pose)
        self.get_logger().info(f"Published initial pose at x:{x}, y:{y}, yaw:{yaw}")

    def joy_callback(self, msg: Joy):
        # Buttons: 0 = resume, 1 = pause
        if len(msg.buttons) > 1:
            if msg.buttons[1] == 1:  # pause
                if not self.paused:
                    self.get_logger().info("Pause button pressed. Pausing mission.")
                    self.pause_mission()
            elif msg.buttons[0] == 1:  # resume
                if self.paused:
                    self.get_logger().info("Resume button pressed. Resuming mission.")
                    self.resume_mission()

    def pause_mission(self):
        self.paused = True
        # Cancel current navigation goal if active
        if self.current_goal_handle is not None and self.goal_active:
            cancel_future = self.current_goal_handle.cancel_goal_async()
            self.get_logger().info("Navigation goal cancellation requested due to pause.")
            self.goal_active = False # Assume goal is no longer active after cancellation request

    def resume_mission(self):
        self.paused = False
        if not self.mission_started:
            self.get_logger().info('Mission starting from the beginning.')
            self.mission_started = True
            self.current_tree_idx = 0
            self.current_point_idx = 0
            # Ensure poses are generated for the first tree immediately after mission starts
            self.poses = self.generate_poses_around_tree(self.tree_positions[self.current_tree_idx])

        self.get_logger().info("Mission resumed.")


    def execute_mission_step(self):
        if self.paused or not self.mission_started:
            # Don't send new goals if paused or mission hasn't explicitly started
            return

        # Only send a new goal if no goal is currently active
        if self.goal_active:
            return

        # Check if mission complete
        if self.current_tree_idx >= len(self.tree_positions):
            self.get_logger().info('Mission complete. You can now save data or end the node.')
            # Stop the timer after mission complete
            self.timer.cancel()
            return

        # If finished points for this tree, move to next tree
        if self.current_point_idx >= len(self.poses):
            self.get_logger().info(f'Finished navigating around {self.tree_positions[self.current_tree_idx]["fruit"]} tree.')
            self.current_tree_idx += 1
            self.current_point_idx = 0
            if self.current_tree_idx < len(self.tree_positions):
                self.get_logger().info(f'Navigating around {self.tree_positions[self.current_tree_idx]["fruit"]} tree.')
                self.poses = self.generate_poses_around_tree(self.tree_positions[self.current_tree_idx])
            else:
                # No more trees, will finish on next loop
                return

        # Send navigation goal for current waypoint
        current_pose = self.poses[self.current_point_idx]
        self.get_logger().info(f'Sending goal {self.current_point_idx + 1}/{len(self.poses)} around {self.tree_positions[self.current_tree_idx]["fruit"]} tree at ({current_pose.pose.position.x:.2f}, {current_pose.pose.position.y:.2f})')

        self.send_nav_goal_async(current_pose)
        self.goal_active = True # Set flag to indicate a goal is in progress

    def generate_poses_around_tree(self, tree):
        poses = []
        for i in range(self.num_points):
            angle = 2 * pi * i / self.num_points
            x = tree['x'] + self.circle_radius * cos(angle)
            y = tree['y'] + self.circle_radius * sin(angle)
            yaw = (angle + pi) % (2 * pi)
            quat = quaternion_from_euler(0, 0, yaw)

            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
            poses.append(pose)
        return poses

    def send_nav_goal_async(self, pose):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose
        self.nav_to_pose_client.send_goal_async(goal_msg).add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle: ClientGoalHandle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected by action server.')
            self.goal_active = False
            self.current_point_idx += 1 # Move to next point as current one was rejected
            return

        self.get_logger().info('Goal accepted.')
        self.current_goal_handle = goal_handle
        goal_handle.get_result_async().add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        status = future.result().status

        self.current_goal_handle = None
        self.goal_active = False

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Goal succeeded!')
            self.trigger_fruit_detection(self.tree_positions[self.current_tree_idx], self.poses[self.current_point_idx])
            self.current_point_idx += 1
        elif status == GoalStatus.STATUS_CANCELED:
            self.get_logger().info('Goal was cancelled.')
            # Do not increment current_point_idx if cancelled, so it retries on resume
        else:
            self.get_logger().warn(f'Goal failed with status code: {status}')
            self.current_point_idx += 1

    def trigger_fruit_detection(self, tree, pose):
        self.get_logger().info(f"Triggering fruit detection at x: {pose.pose.position.x:.2f}, y: {pose.pose.position.y:.2f} for {tree['fruit']} tree.")
        # TODO: Add detection logic

def main(args=None):
    rclpy.init(args=args)
    navigator = TreeNavigator()
    rclpy.spin(navigator)
    navigator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()