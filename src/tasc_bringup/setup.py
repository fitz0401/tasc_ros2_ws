from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'tasc_bringup'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    py_modules=['tasc_node', 'joy_listener', 'websocket_server'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=[
        'setuptools',
        'rclpy',
        'geometry_msgs',
        'sensor_msgs',
        'std_msgs',
        'tf2_ros',
        'tf2_geometry_msgs',
        'moveit_msgs',
        'control_msgs',
        'numpy',
        'cv_bridge',
        'image_geometry',
        'message_filters',
        'opencv-python',
        'Pillow',
        'scipy',
        'websockets',
        'asyncio',
        'sensor_msgs_py',
    ],
    zip_safe=True,
    maintainer='u0177383',
    maintainer_email='u0177383@kuLeuven.be',
    description='TASC Bringup Package with WebSocket Server',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tasc_node = tasc_node:main',
        ],
    },
)
