# Adaptive ResNet Controller for QuadSim

Online adaptive deep residual neural network for drone velocity control using ROS2. Real-time learning and adaptation for quadrotor control in Unreal Engine simulation.

## Features

- **Online Training**: Real-time weight updates during flight
- **Safety Mechanisms**: Multiple safety checks and emergency stop
- **Adaptive Learning**: Learning rate decay and error-based training control
- **ROS2 Integration**: Full integration with QuadSim topics and services
- **Diagnostics**: Real-time performance monitoring and status reporting

## Installation

1. Clone into your ROS2 workspace:
   ```bash
   cd ~/ros2_ws/src
   git clone <this_repo> adaptive_resnet_controller
   ```

2. Install Python dependencies:
   ```bash
   cd adaptive_resnet_controller
   pip install -r requirements.txt
   ```

3. Build the ROS2 package:
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select adaptive_resnet_controller
   source install/setup.bash
   ```

## Usage

### Basic Launch
```bash
ros2 launch adaptive_resnet_controller adaptive_controller.launch.py
```

### Launch with Training Enabled
```bash
ros2 launch adaptive_resnet_controller adaptive_controller.launch.py enable_training:=true
```

### Enable/Disable Training at Runtime
```bash
ros2 service call /resnet/enable_training std_srvs/srv/SetBool "{data: true}"
```

### Reset Neural Network Weights
```bash
ros2 service call /resnet/reset_weights std_srvs/srv/Trigger
```

## Architecture

### Neural Network
- Deep Residual Network (ResNet) with configurable blocks and layers
- Input: Position (3), Velocity (3), Reference Velocity (3), Error (3), Orientation (3)
- Output: Velocity corrections [Δvx, Δvy, Δvz]
- Online least-squares weight updates

### ROS2 Topics

**Subscriptions:**
- `/quadsim/odom` (nav_msgs/Odometry): Current drone state
- `/quadsim/imu` (sensor_msgs/Imu): IMU data
- `/ref/vel_local` (geometry_msgs/TwistStamped): Reference velocity
- `/baseline/cmd_vel` (geometry_msgs/TwistStamped): Optional baseline commands

**Publications:**
- `/resnet/cmd_vel_residual` (geometry_msgs/TwistStamped): Velocity corrections
- `/resnet/diagnostics` (std_msgs/Float32MultiArray): Performance metrics
- `/resnet/status` (std_msgs/String): Controller status

### ROS2 Services
- `/resnet/enable_training` (std_srvs/SetBool): Enable/disable online training
- `/resnet/reset_weights` (std_srvs/Trigger): Reset neural network weights

## Configuration

Key parameters in `config/adaptive_controller_params.yaml`:

- `resnet.num_blocks`: Number of residual blocks (default: 2)
- `resnet.num_neurons`: Neurons per layer (default: 20)
- `learning.initial_rate`: Initial learning rate (default: 0.001)
- `control.max_velocity_correction`: Maximum velocity correction (default: 0.5 m/s)
- `safety.emergency_stop_error`: Emergency stop threshold (default: 5.0 m/s)

## Integration with QuadSim

The controller integrates seamlessly with your Unreal QuadSim setup:

1. **C++ Side (AROS2Controller)**: Publishes odometry and subscribes to ResNet corrections
2. **Python Side (This Package)**: Computes adaptive corrections based on tracking error
3. **Control Flow**:
   - QuadSim → Odometry → ResNet → Corrections → QuadSim

## Safety Features

- **Warmup Period**: No training for first 100 steps
- **Error Thresholds**: Training disabled if error too small or too large
- **Output Limiting**: Maximum correction magnitude enforced
- **Emergency Stop**: Controller disables if error exceeds safety threshold
- **Thread Safety**: Proper locking for multi-threaded execution

## Monitor Status

```bash
ros2 topic echo /resnet/status
ros2 topic echo /resnet/diagnostics
```

## Troubleshooting

1. **No odometry data**: Check if QuadSim is running and publishing to `/quadsim/odom`
2. **Training not converging**: Adjust learning rate or network architecture
3. **Emergency stops**: Reduce `max_error_threshold` or check reference commands
4. **High CPU usage**: Reduce `control.rate_hz` or use fewer neurons

## Based On

This implementation is based on the paper:

```
@article{Nino.Patil.ea2025,
  author  = {Cristian F. Nino and Omkar Sudhir Patil and Marla R. Eisman and Warren E. Dixon},
  title   = {Online ResNet-Based Adaptive Control for Nonlinear Target Tracking},
  year    = {2025},
  journal = {IEEE Control Systems Letters},
  volume  = {9},
  pages   = {907-912},
  doi     = {10.1109/LCSYS.2025.3576652}
}
```
