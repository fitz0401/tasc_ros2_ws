from sensor_msgs.msg import Joy
from geometry_msgs.msg import TwistStamped

# Define enum-like constants
LEFT_STICK_X = 0
LEFT_STICK_Y = 1
LEFT_TRIGGER = 2
RIGHT_STICK_X = 3
RIGHT_STICK_Y = 4
RIGHT_TRIGGER = 5
D_PAD_X = 6
D_PAD_Y = 7

A = 0
B = 1
X = 2
Y = 3
LEFT_BUMPER = 4
RIGHT_BUMPER = 5
CHANGE_VIEW = 6
MENU = 7
HOME = 8
LEFT_STICK_CLICK = 9
RIGHT_STICK_CLICK = 10

AXIS_DEFAULTS = {
    LEFT_TRIGGER: 1.0,
    RIGHT_TRIGGER: 1.0,
}


class JoyListener:
    def __init__(self):
        self.latest_axes = []
        self.latest_buttons = []

    def update_from_joy_msg(self, msg: Joy):
        self.latest_axes = list(msg.axes)
        self.latest_buttons = list(msg.buttons)

    def get_twist(self, stamp, frame_id="fr3_link0"):
        if not self.latest_axes or not self.latest_buttons:
            return None

        axes = self.latest_axes
        buttons = self.latest_buttons

        if len(axes) < 8 or len(buttons) < 11:
            print("JoyListener: Not enough axes or buttons received.")
            return None

        twist = TwistStamped()
        twist.header.stamp = stamp
        twist.header.frame_id = frame_id

        # Map axes to twist commands
        twist.twist.linear.x = -float(axes[RIGHT_STICK_X])
        twist.twist.linear.y = float(axes[RIGHT_STICK_Y])
        lin_z_trigger = 0.4 * (float(axes[RIGHT_TRIGGER]) - AXIS_DEFAULTS[RIGHT_TRIGGER])   # Down
        lin_z_bumper = 0.8 * float(buttons[RIGHT_BUMPER])   # Up
        twist.twist.linear.z = lin_z_trigger + lin_z_bumper
        
        twist.twist.angular.x = -float(axes[LEFT_STICK_X])
        twist.twist.angular.y = float(axes[LEFT_STICK_Y])
        ang_z_trigger = 0.4 * (float(axes[LEFT_TRIGGER]) - AXIS_DEFAULTS[LEFT_TRIGGER])
        ang_z_bumper = 0.8 * float(buttons[LEFT_BUMPER])
        twist.twist.angular.z = ang_z_trigger + ang_z_bumper

        return twist

    def get_gripper_command(self):
        """Returns: 'close' / 'open' / None"""
        if not self.latest_buttons:
            return None

        if self.latest_buttons[A]:
            return 'close'
        elif self.latest_buttons[B]:
            return 'open'
        else:
            return None
