# Moveit Servo

See the [Realtime Arm Servoing Tutorial](https://moveit.picknik.ai/main/doc/realtime_servo/realtime_servo_tutorial.html) for installation instructions, quick-start guide, an overview about `moveit_servo`, and to learn how to set it up on your robot.

# Teleoperation


```
## Home
ros2 launch franka_bringup move_to_start_example_controller.launch.py robot_ip:=172.16.0.2
```
```
## Lower
ros2 launch franka_fr3_moveit_config lower.launch.py robot_ip:=172.16.0.2
```
```
## Teleop
ros2 launch moveit_servo teleop.launch.py
```