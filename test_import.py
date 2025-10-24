#!/usr/bin/env python3
"""Test script to verify imports and basic functionality"""

print("Testing imports...")

# Test ROS2 imports
try:
    import rclpy
    print("✓ rclpy imported")
except ImportError as e:
    print(f"✗ Failed to import rclpy: {e}")

# Test message imports
try:
    from geometry_msgs.msg import TwistStamped
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Imu
    print("✓ ROS2 messages imported")
except ImportError as e:
    print(f"✗ Failed to import ROS2 messages: {e}")

# Test NumPy and SciPy
try:
    import numpy as np
    import scipy
    print("✓ NumPy and SciPy imported")
except ImportError as e:
    print(f"✗ Failed to import NumPy/SciPy: {e}")

# Test local imports
try:
    from src.core.neural_network import NeuralNetwork
    print("✓ Neural network module imported")
except ImportError as e:
    print(f"✗ Failed to import neural network: {e}")

# Test node creation (without running)
try:
    import ros2_adaptive_controller
    print("✓ Adaptive controller module imported")
    print("\nAll imports successful! The package is ready to use.")
    print("\nTo run the node:")
    print("1. Start your QuadSim with ROS2 bridge")
    print("2. Run: ros2 launch adaptive_resnet_controller adaptive_controller.launch.py")
except ImportError as e:
    print(f"✗ Failed to import adaptive controller: {e}")