import rclpy
import message_filters
import tf2_ros
import tf2_geometry_msgs
import numpy as np

from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import Image, Joy, CameraInfo
from geometry_msgs.msg import TwistStamped, PoseStamped
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
from std_srvs.srv import Trigger
from control_msgs.action import GripperCommand, FollowJointTrajectory
from franka_msgs.action import Grasp, Move
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from joy_listener import JoyListener
from websocket_server import AANPWebSocketServer


class AANPMain(Node):
    def __init__(self):
        super().__init__('aanp_main_node')

        # Initialize TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Initialize WebSocket server
        self.websocket_server = AANPWebSocketServer(
            host="0.0.0.0", 
            port=8765, 
            logger=self.get_logger(),
            max_clients=2,
            tf_buffer=self.tf_buffer,  # Pass TF buffer for coordinate transformations
            target_frame="fr3_link0",  # Target frame for point cloud
            camera_frame_id="camera_color_optical_frame"  # Use color optical frame for aligned depth
        )
        
        # Set up assist action callback
        self.websocket_server.set_assist_action_callback(self.handle_assist_action)
        
        # Start WebSocket server
        self.websocket_server.start_server()
        self.get_logger().info("WebSocket server started on port 8765")
        
        # Variables for assistance
        self.assist_action = np.zeros(6)  # [x, y, z, roll, pitch, yaw]
        self.assist_action_received = False

        # Gripper button state tracking to prevent multiple commands
        self.gripper_button_a_pressed = False  # Track A button (close gripper)
        self.gripper_button_b_pressed = False  # Track B button (open gripper)
        self.reset_button_x_pressed = False    # Track X button (robot reset)
        self.servo_restart_button_y_pressed = False # Track Y button (restart servo)
        
        # Robot reset state tracking
        self.robot_reset_in_progress = False

        # sync depth and RGB image topics
        # Use aligned depth for perfect pixel correspondence with RGB
        depth_sub = message_filters.Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw')
        image_sub = message_filters.Subscriber(self, Image, '/camera/color/image_raw')
        ts = message_filters.ApproximateTimeSynchronizer(
            [depth_sub, image_sub], queue_size=10, slop=0.1
        )
        ts.registerCallback(self.sensor_callback)
        self.got_frame = False
        self.latest_depth = None
        self.latest_image = None

        ## Subscribers
        # 1. TF2 buffer and listener for getting EEF pose
        self.current_eef_pose = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 2. Joy listener for joystick input
        self.joy_listener = JoyListener()
        self.joy_sub = self.create_subscription(Joy, '/joy', self.joy_callback, 10)

        # 3. Subscribe to camera info to get accurate intrinsics
        # Use aligned depth camera info for perfect correspondence with RGB
        self.camera_info_sub = self.create_subscription(
            CameraInfo, 
            '/camera/aligned_depth_to_color/camera_info',  # Use aligned depth camera info
            self.camera_info_callback, 
            10
        )
        self.camera_info_received = False

        ## Publisher
        self.twist_pub = self.create_publisher(TwistStamped, '/servo_node/delta_twist_cmds', 10)

        # 1. Servo service client - start servo node
        self.servo_start_client = self.create_client(Trigger, '/servo_node/start_servo')
        self.servo_pause_client = self.create_client(Trigger, '/servo_node/pause_servo')
        self.servo_unpause_client = self.create_client(Trigger, '/servo_node/unpause_servo')
        # Start servo service
        self.servo_start_timer = self.create_timer(1.0, self.start_servo_service)  # Delay 1 second before starting

        # 2. Gripper action client - both standard GripperCommand and Franka-specific actions
        self.gripper_client = ActionClient(self, GripperCommand, '/fr3_gripper/gripper_action')
        self.franka_grasp_client = ActionClient(self, Grasp, '/fr3_gripper/grasp')
        self.franka_move_client = ActionClient(self, Move, '/fr3_gripper/move')
        self.joint_trajectory_client = ActionClient(self, FollowJointTrajectory, '/fr3_arm_controller/follow_joint_trajectory')

        # 3. Publish twist commands at a regular interval
        self.twist_pub_timer = self.create_timer(0.1, self.twist_timer_callback)

    def handle_assist_action(self, assist_action):
        """Handle assistance action received from WebSocket client"""
        self.assist_action = np.array(assist_action)
        self.assist_action_received = True
        self.get_logger().debug(f"Received assist action: {assist_action}")

    def sensor_callback(self, depth_msg, image_msg):
        if not self.got_frame:
            self.got_frame = True
            self.get_logger().info("Received first frame: Depth and RGB Image. Starting AANP logic...")
            
        self.latest_depth = depth_msg
        self.latest_image = image_msg
        
        # Store raw sensor data for on-demand processing
        self.websocket_server.store_raw_sensor_data(depth_msg, image_msg)
        self.get_logger().debug(f"Sensor callback triggered - stored raw sensor data for on-demand processing")
    
    def joy_callback(self, msg):
        # Update joystick listener
        self.joy_listener.update_from_joy_msg(msg)
        
        # Gripper control - only send command on button press (not hold)
        gripper_action = 0
        
        # A button: close gripper using Franka's Grasp action for better grip
        if msg.buttons[0]:  # A button pressed
            gripper_action = 1  # Close
            if not self.gripper_button_a_pressed:
                # Use Franka's grasp action for continuous gripping force
                # Increased force and smaller width for better grip on square objects
                self.send_franka_grasp_command(width=0.01, force=30.0, speed=0.04)
                
                # Alternative: Use standard gripper command (uncomment if needed)
                # self.send_gripper_command(0.0, 20.0)
                
                self.gripper_button_a_pressed = True
        else:
            self.gripper_button_a_pressed = False  # Reset when button released

        # B button: open gripper using Franka's Move action
        if msg.buttons[1]:  # B button pressed
            gripper_action = -1  # Open
            if not self.gripper_button_b_pressed:
                self.send_franka_move_command(width=0.08, speed=0.05)
                self.gripper_button_b_pressed = True
        else:
            self.gripper_button_b_pressed = False  # Reset when button released

        # X button: reset robot to home position (usually button index 3)
        if len(msg.buttons) > 3 and msg.buttons[3]:  # X button pressed
            if not self.reset_button_x_pressed:
                self.send_robot_to_home_position()
                self.reset_button_x_pressed = True
        else:
            self.reset_button_x_pressed = False  # Reset when button released

        # Y button: manually restart servo after robot reset (usually button index 2)
        if len(msg.buttons) > 2 and msg.buttons[2]:  # Y button pressed
            if not self.servo_restart_button_y_pressed:
                self.restart_servo_manually()
                self.servo_restart_button_y_pressed = True
        else:
            self.servo_restart_button_y_pressed = False  # Reset when button released

        self.websocket_server.update_gripper_action(gripper_action)

    def twist_timer_callback(self):
        if not self.got_frame:
            return
            
        # Get the end-effector pose
        eef_pose = self.get_eef_pose()
        if not eef_pose:
            self.get_logger().warn("EEF pose not available, skipping twist command.")
            return

        # Get human twist input once to ensure data consistency
        current_twist = self.joy_listener.get_twist(self.get_clock().now().to_msg())

        # # Teleoperation Controller
        # vel_x, vel_y, vel_z, vel_roll, vel_pitch, vel_yaw = self.teleop_controller(eef_pose, current_twist)

        # Shared Controller
        vel_x, vel_y, vel_z, vel_roll, vel_pitch, vel_yaw = self.shared_controller(eef_pose, current_twist)

        # Send twist command
        twist = TwistStamped()
        twist.header.stamp = self.get_clock().now().to_msg()
        twist.header.frame_id = "fr3_link0"
        twist.twist.linear.x = vel_x
        twist.twist.linear.y = vel_y
        twist.twist.linear.z = vel_z
        twist.twist.angular.x = vel_roll
        twist.twist.angular.y = vel_pitch
        twist.twist.angular.z = vel_yaw
        self.twist_pub.publish(twist)

        # Debug information
        if any([vel_x, vel_y, vel_z, vel_roll, vel_pitch, vel_yaw]):
            self.get_logger().debug(
                f"send twist to: fr3_link0: "
                f"linear=({vel_x:.3f}, {vel_y:.3f}, {vel_z:.3f}), "
                f"angular=({vel_roll:.3f}, {vel_pitch:.3f}, {vel_yaw:.3f})"
            )

    def get_eef_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                'fr3_link0',
                'fr3_hand',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            # transform -> PoseStamped
            pose = PoseStamped()
            pose.header = transform.header
            pose.header.frame_id = 'fr3_link0'
            pose.pose.position.x = transform.transform.translation.x
            pose.pose.position.y = transform.transform.translation.y
            pose.pose.position.z = transform.transform.translation.z
            pose.pose.orientation = transform.transform.rotation
            
            self.current_eef_pose = pose
            return pose
            
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f"Can not get EEF pose: {e}")
            return None

    def teleop_controller(self, eef_pose, human_twist):
        """Pure teleoperation controller"""
        if human_twist is None:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        return human_twist.twist.linear.x, human_twist.twist.linear.y, human_twist.twist.linear.z, \
               human_twist.twist.angular.x, human_twist.twist.angular.y, human_twist.twist.angular.z
    
    def shared_controller(self, eef_pose, human_twist):
        """Shared controller combining human input and AI assistance"""
        # Initialize velocities
        vel_x, vel_y, vel_z = 0.0, 0.0, 0.0
        vel_roll, vel_pitch, vel_yaw = 0.0, 0.0, 0.0
        
        # Add human input
        if human_twist is not None:
            vel_x += human_twist.twist.linear.x
            vel_y += human_twist.twist.linear.y
            vel_z += human_twist.twist.linear.z
            vel_roll += human_twist.twist.angular.x
            vel_pitch += human_twist.twist.angular.y
            vel_yaw += human_twist.twist.angular.z
            
        # Update WebSocket server with the same twist data being used
        if self.websocket_server.get_connected_clients_count() > 0:
            # Update EEF pose
            self.websocket_server.update_eef_pose(eef_pose)
            
            # Update twist data with the same data we're using for control
            linear = np.array([vel_x, vel_y, vel_z])
            angular = np.array([vel_roll, vel_pitch, vel_yaw])
            self.websocket_server.update_twist(linear, angular)
            
            # Publish all data to clients
            self.websocket_server.publish_all_data()
        # When assistance action is received (self.assist_action_received will be updated in callback function), 
        # apply the assistance
        if self.assist_action_received:
            self.get_logger().debug(f"Applying assist action: {self.assist_action}")
            assist_scale = 1.5  # Adjust this value as needed
            vel_x += self.assist_action[0] * assist_scale
            vel_y += self.assist_action[1] * assist_scale
            vel_z += self.assist_action[2] * assist_scale
            vel_roll += self.assist_action[3] * assist_scale
            vel_pitch += self.assist_action[4] * assist_scale
            vel_yaw += self.assist_action[5] * assist_scale
            # Reset assistance flag
            self.assist_action_received = False
            
        return vel_x, vel_y, vel_z, vel_roll, vel_pitch, vel_yaw

    def start_servo_service(self):
        if self.servo_start_client.service_is_ready():
            request = Trigger.Request()
            future = self.servo_start_client.call_async(request)
            self.get_logger().info("Servo service starting...")
            self.servo_start_timer.cancel()  # Cancel the timer after the first call
        else:
            self.get_logger().warn("Servo service not ready, waiting...")

    def send_gripper_command(self, position, max_effort):
        """Standard gripper command - for backward compatibility"""
        if not self.gripper_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn("Gripper action server not available")
            return
        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = position
        goal_msg.command.max_effort = max_effort
        self.gripper_client.send_goal_async(goal_msg)
        self.get_logger().info(f"Standard gripper command sent: position={position}, effort={max_effort}")
    
    def send_franka_grasp_command(self, width=0.01, force=30.0, speed=0.03):
        """
        Franka-specific grasp command that maintains grip force
        Args:
            width (float): Target grasp width in meters (0.005m = 5mm gap for tight grip)
            force (float): Continuous grasping force in Newtons (increased to 50N)
            speed (float): Grasping speed in m/s (slightly slower for better control)
        """
        if not self.franka_grasp_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn("Franka grasp action server not available")
            return
            
        goal_msg = Grasp.Goal()
        goal_msg.width = width  # Target gap between fingers
        goal_msg.force = force  # Force to maintain
        goal_msg.speed = speed  # Speed of closing
        
        # Set tighter grasp tolerances for better grip
        goal_msg.epsilon.inner = 0.002  # Tighter inner tolerance (was 0.005)
        goal_msg.epsilon.outer = 0.002  # Tighter outer tolerance (was 0.005)
        
        future = self.franka_grasp_client.send_goal_async(goal_msg)
        self.get_logger().info(f"Franka grasp command sent: width={width}m, force={force}N, speed={speed}m/s")
        self.get_logger().info(f"Using tight grip settings for secure object holding")
    
    def send_franka_move_command(self, width=0.08, speed=0.05):
        """
        Franka-specific move command for opening gripper
        Args:
            width (float): Target width in meters (0.08m = 8cm fully open)
            speed (float): Movement speed in m/s
        """
        if not self.franka_move_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn("Franka move action server not available")
            return
            
        goal_msg = Move.Goal()
        goal_msg.width = width
        goal_msg.speed = speed
        
        future = self.franka_move_client.send_goal_async(goal_msg)
        self.get_logger().info(f"Franka move command sent: width={width}m, speed={speed}m/s")
    
    def send_robot_to_home_position(self):
        """
        Send robot to home/start position using FollowJointTrajectory action
        Home position: [0, -π/4, 0, -3π/4, 0, π/2, π/4]
        Based on franka_example_controllers::MoveToStartExampleController
        """
        try:
            import math
            
            # First pause servo to release joint control
            if self.servo_pause_client.service_is_ready():
                request = Trigger.Request()
                future = self.servo_pause_client.call_async(request)
                self.get_logger().info("Pausing servo node before sending trajectory")
            
            # Set reset in progress flag
            self.robot_reset_in_progress = True
            
            # Check if action server is available
            if not self.joint_trajectory_client.wait_for_server(timeout_sec=2.0):
                self.get_logger().warn("Joint trajectory action server not available")
                return
            
            # Define home position (same as in the C++ controller)
            home_position = [
                0.0,                    # joint1
                -math.pi / 4,          # joint2: -π/4 = -0.785
                0.0,                    # joint3
                -3 * math.pi / 4,      # joint4: -3π/4 = -2.356
                0.0,                    # joint5
                math.pi / 2,           # joint6: π/2 = 1.571
                math.pi / 4            # joint7: π/4 = 0.785
            ]
            
            # Create FollowJointTrajectory goal
            goal_msg = FollowJointTrajectory.Goal()
            
            # Set up trajectory
            trajectory_msg = JointTrajectory()
            trajectory_msg.header.stamp.sec = 0  # Use 0 timestamp for immediate execution
            trajectory_msg.header.stamp.nanosec = 0
            trajectory_msg.header.frame_id = ""  # Empty frame_id as per working example
            
            # Set joint names for FR3
            trajectory_msg.joint_names = [
                'fr3_joint1', 'fr3_joint2', 'fr3_joint3', 'fr3_joint4',
                'fr3_joint5', 'fr3_joint6', 'fr3_joint7'
            ]
            
            # Create trajectory point
            point = JointTrajectoryPoint()
            point.positions = home_position
            point.velocities = [0.0] * 7  # Stop at target
            point.time_from_start = Duration(sec=6, nanosec=0)  # 6 seconds to reach target

            trajectory_msg.points = [point]
            goal_msg.trajectory = trajectory_msg
            
            # Send goal and wait for response
            future = self.joint_trajectory_client.send_goal_async(goal_msg)
            self.get_logger().info("Sending robot to home position: [0, -π/4, 0, -3π/4, 0, π/2, π/4]")
            self.get_logger().info("Servo will be paused during reset. Press Y button to restart servo after reset completes.")
            
            # Add callback to check if goal was accepted
            future.add_done_callback(self.home_position_goal_callback)
            
        except Exception as e:
            self.get_logger().error(f"Failed to send robot to home position: {e}")
    
    def home_position_goal_callback(self, future):
        """Callback to check if home position goal was accepted"""
        try:
            goal_handle = future.result()
            if goal_handle.accepted:
                self.get_logger().info("Home position goal accepted, robot should be moving")
                # Add result callback to unpause servo when trajectory is complete
                get_result_future = goal_handle.get_result_async()
                get_result_future.add_done_callback(self.home_position_result_callback)
            else:
                self.get_logger().warn("Home position goal rejected")
        except Exception as e:
            self.get_logger().error(f"Home position goal failed: {e}")
    
    def home_position_result_callback(self, future):
        """Callback when home position trajectory is complete"""
        try:
            result = future.result()
            self.get_logger().info(f"Home position trajectory completed with result: {result.result.error_code}")
            self.get_logger().info("✓ Robot reset complete. Press Y button to restart servo when ready.")
            
            # Clear reset in progress flag
            self.robot_reset_in_progress = False
            
        except Exception as e:
            self.get_logger().error(f"Home position result callback failed: {e}")
            self.robot_reset_in_progress = False
    
    def restart_servo_manually(self):
        """Manually restart servo - called by Y button"""
        try:
            if self.robot_reset_in_progress:
                self.get_logger().warn("Robot reset still in progress. Please wait until reset completes.")
                return
                
            if self.servo_unpause_client.service_is_ready():
                request = Trigger.Request()
                future = self.servo_unpause_client.call_async(request)
                self.get_logger().info("✓ Manually restarting servo node (Y button pressed)")
                self.get_logger().info("Robot is now ready for teleoperation.")
            else:
                self.get_logger().warn("Servo unpause service not ready")
                
        except Exception as e:
            self.get_logger().error(f"Manual servo restart failed: {e}")
    
    def camera_info_callback(self, msg):
        """Update camera intrinsics from CameraInfo message"""
        if not self.camera_info_received:
            self.websocket_server.update_camera_intrinsics_from_camera_info(msg)
            self.camera_info_received = True
            self.get_logger().info("Updated camera intrinsics from camera_info topic")

    def destroy_node(self):
        """Clean up WebSocket server when node is destroyed"""
        self.get_logger().info("Shutting down WebSocket server...")
        self.websocket_server.stop_server()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = AANPMain()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received, shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
