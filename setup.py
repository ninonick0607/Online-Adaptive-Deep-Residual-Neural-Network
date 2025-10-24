from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'adaptive_resnet_controller'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@example.com',
    description='Online Adaptive ResNet Controller for QuadSim',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'adaptive_controller = ros2_adaptive_controller:main',
        ],
    },
)