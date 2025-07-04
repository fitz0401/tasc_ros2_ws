from setuptools import setup
import os
from glob import glob

package_name = 'aanp_bringup'

setup(
    name=package_name,
    version='0.0.0',
    packages=[],
    py_modules=['aanp_node', 'joy_listener'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='u0177383',
    maintainer_email='u0177383@kuLeuven.be',
    description='AANP Bringup Package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'aanp_node = aanp_node:main',
        ],
    },
)
