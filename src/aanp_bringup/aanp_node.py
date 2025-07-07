import rclpy
import message_filters
import tf2_ros
import tf2_geometry_msgs

from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import PointCloud2, Image, Joy
from geometry_msgs.msg import TwistStamped, PoseStamped
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
from std_srvs.srv import Trigger
from control_msgs.action import GripperCommand
from joy_listener import JoyListener


class AANPMain(Node):
    def __init__(self):
        super().__init__('aanp_main_node')

        # sync pointcloud and image topics
        point_sub = message_filters.Subscriber(self, PointCloud2, '/camera/depth/color/points')
        image_sub = message_filters.Subscriber(self, Image, '/camera/color/image_raw')
        ts = message_filters.ApproximateTimeSynchronizer(
            [point_sub, image_sub], queue_size=10, slop=0.1
        )
        ts.registerCallback(self.sensor_callback)
        self.got_frame = False
        self.latest_pointcloud = None
        self.latest_image = None

        ## Subscribers
        # 1. TF2 buffer and listener for getting EEF pose
        self.current_eef_pose = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 2. Joy listener for joystick input
        self.joy_listener = JoyListener()
        self.joy_sub = self.create_subscription(Joy, '/joy', self.joy_callback, 10)

        ## Publisher
        self.twist_pub = self.create_publisher(TwistStamped, '/servo_node/delta_twist_cmds', 10)

        # 1. Servo service client - start servo node
        self.servo_start_client = self.create_client(Trigger, '/servo_node/start_servo')
        # Start servo service
        self.servo_start_timer = self.create_timer(1.0, self.start_servo_service)  # Delay 1 second before starting

        # 2. Gripper action client
        self.gripper_client = ActionClient(self, GripperCommand, '/fr3_gripper/gripper_action')

        # 3. Publish twist commands at a regular interval
        self.twist_pub_timer = self.create_timer(0.1, self.twist_timer_callback)
        

    def sensor_callback(self, pointcloud_msg, image_msg):
        if not self.got_frame:
            self.got_frame = True
            self.get_logger().info("Received first frame: PointCloud and Image. Starting AANP logic...")
            # Send the latest pointcloud and image to the WebSocket server once
            # ...
        self.latest_pointcloud = pointcloud_msg
        self.latest_image = image_msg
    
    def joy_callback(self, msg):
        self.joy_listener.update_from_joy_msg(msg)
        # Gripper control
        if msg.buttons[0]:  # A：close gripper
            self.send_gripper_command(0.0, 20.0)
        elif msg.buttons[1]:  # B：open gripper
            self.send_gripper_command(0.039, 20.0)
        # Send the gripper command to the WebSocket server
        # ...

    def twist_timer_callback(self):
        if not self.got_frame:
            return
            
        # Get the end-effector pose
        eef_pose = self.get_eef_pose()
        if not eef_pose:
            self.get_logger().warn("EEF pose not available, skipping twist command.")
            return

        # # Teleoperation Controller
        vel_x, vel_y, vel_z, vel_roll, vel_pitch, vel_yaw = self.teleop_controller(eef_pose)

        # Shared Controller
        # vel_x, vel_y, vel_z, vel_roll, vel_pitch, vel_yaw = self.shared_controller(eef_pose)

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

    def teleop_controller(self, eef_pose):
        twist_msg = self.joy_listener.get_twist(self.get_clock().now().to_msg())
        if twist_msg is None:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        return twist_msg.twist.linear.x, twist_msg.twist.linear.y, twist_msg.twist.linear.z, \
               twist_msg.twist.angular.x, twist_msg.twist.angular.y, twist_msg.twist.angular.z
    
    def shared_controller(self, eef_pose):
        twist_msg = self.joy_listener.get_twist(self.get_clock().now().to_msg())
        # Send the twist info to the WebSocket server
        # ...

        # Receive the assist action from the WebSocket server
        # ...

        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    def start_servo_service(self):
        if self.servo_start_client.service_is_ready():
            request = Trigger.Request()
            future = self.servo_start_client.call_async(request)
            self.get_logger().info("Servo service starting...")
            self.servo_start_timer.cancel()  # Cancel the timer after the first call
        else:
            self.get_logger().warn("Servo service not ready, waiting...")

    def send_gripper_command(self, position, max_effort):
        if not self.gripper_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn("Gripper action server not available")
            return
        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = position
        goal_msg.command.max_effort = max_effort
        self.gripper_client.send_goal_async(goal_msg)
        self.get_logger().info(f"Gripper command sent: position={position}, effort={max_effort}")

def main(args=None):
    rclpy.init(args=args)
    node = AANPMain()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
