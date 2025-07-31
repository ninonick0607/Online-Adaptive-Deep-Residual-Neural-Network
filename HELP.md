# HELP.md - Technical Documentation

## Program Architecture and Logic

### System Overview

This framework implements online adaptive control using deep residual neural networks (ResNets) for nonlinear dynamical system control. The system performs real-time target tracking where neural networks continuously adapt their weights based on tracking error feedback using online learning algorithms.

The architecture employs a modular design separating concerns into distinct functional areas: neural network computation, system dynamics modeling, numerical integration, data management, and visualization.

### Core Architecture Components

#### Neural Network Module (`src/core/neural_network.py`)

**Class: `NeuralNetwork`**

Implements a deep residual neural network with online adaptive learning capabilities.

**Key Attributes:**
- `weights`: Network weight matrix (NDArray[np.float64])
- `learning_rate`: Adaptive learning rate matrix for online updates
- `neural_network_gradient_wrt_weights`: Gradient computation storage
- `alpha`, `beta`, `gamma`: Learning rate adaptation parameters derived from singular value bounds

**Key Methods:**
- `__init__(input_func, config)`: Initializes network architecture and learning parameters
- `initialize_weights()`: Xavier/He initialization based on activation functions
- `generate_initialized_weights(input_size, output_size, variance)`: Weight initialization with variance scaling
- `train_step(step, loss)`: Online learning update using tracking error
- `forward_pass(input_vector)`: Forward propagation through residual blocks
- `backward_pass_gradient(input_vector)`: Computes gradients for weight updates

**Residual Architecture:**
The network implements residual connections (shortcuts) that allow gradients to flow directly through the network, preventing vanishing gradient problems in deeper architectures. Each residual block contains:
- Input layer → Hidden layers → Output layer
- Shortcut connection from input directly to output
- Element-wise addition of main path and shortcut

**Online Learning Algorithm:**
Uses a modified recursive least squares approach with forgetting factors for continuous weight adaptation. The learning rate adapts based on the Hessian approximation to ensure stable convergence.

#### Entity Module (`src/core/entity.py`)

**Base Class: `Entity`**
- `positions`: State trajectory storage (NDArray[np.float64])
- `velocities`: Velocity trajectory storage
- `num_states`: System dimensionality
- `time_step_delta`: Integration time step

**Class: `Agent(Entity)`**

Represents a controlled agent with neural network-based adaptive control.

**Key Attributes:**
- `target`: Reference to target entity for tracking
- `neural_network`: Embedded NeuralNetwork instance
- `control_output`: Computed control signal
- `tracking_error`: Current tracking error
- `k1`: Proportional control gain

**Key Methods:**
- `compute_control_output(step)`: Computes control law combining proportional and neural network terms
- `update_dynamics(step)`: Updates agent state using numerical integration
- `_input_func(step)`: Defines neural network input (target position)

**Control Law:**
```
u(t) = k1 * e(t) + NN(target_position)
```
where e(t) is tracking error and NN() is the neural network output.

**Class: `Target(Entity)`**

Represents the reference trajectory to be tracked. Follows autonomous dynamics without control input.

#### Simulation Module (`src/simulation/`)

**Dynamics Module (`dynamics.py`)**

Implements mathematical models for various dynamical systems:

**1. Attitude MRP (`attitude_mrp`)**
- **Purpose**: Spacecraft attitude control using Modified Rodrigues Parameters
- **State Variables**: [r1, r2, r3] (unitless MRP representation)
- **Physics**: Rigid body attitude kinematics with constant angular velocity
- **Equations**: r_dot = 0.5 * B(r) * ω where B(r) is the kinematic matrix

**2. Chua Circuit (`chua`)**
- **Purpose**: Chaotic double-scroll circuit dynamics
- **State Variables**: [x, y, z] (normalized capacitor voltages and inductor current)
- **Parameters**: α=15.6, β=33, m0=-1.143, m1=-0.714
- **Behavior**: Exhibits chaotic behavior with double-scroll attractor

**3. Trophic Dynamics (`trophic_dynamics`)**
- **Purpose**: Three-tier ecological food chain model
- **State Variables**: [H, P, T] (herbivore, predator, top-predator populations)
- **Parameters**: Growth rates, carrying capacity, predation rates, mortality rates
- **Equations**: Lotka-Volterra type predator-prey dynamics

**4. Custom (`custom`)**
- **Purpose**: Placeholder for user-defined dynamics
- **Implementation**: Returns zero derivatives (stable equilibrium)

**Integration Module (`integrate.py`)**

**Function: `integrate_step`**
- Implements fourth-order Runge-Kutta (RK4) numerical integration
- Fixed time step integration for deterministic evolution
- Handles control input integration for controlled systems

#### Data Management Module (`src/io/data_manager.py`)

**Key Functions:**
- `save_state_to_csv`: Logs agent/target positions, velocities, tracking errors
- `save_nn_to_csv`: Logs neural network weights and outputs
- `close_all_files`: Ensures proper file closure and data persistence

**Implementation Details:**
- Uses buffered I/O for performance optimization (buffer size: 100 entries)
- Automatic file handle management with cleanup
- Separate CSV files for each agent type and neural network data
- Progress tracking during simulation execution

#### Visualization Module (`src/visualization/plotter.py`)

**Key Functions:**
- `results()`: Main plotting interface
- `configure_plot()`: IEEE standard formatting
- `get_simulation_data()`: Data loading from CSV files
- `plot_tracking_error()`: Error analysis visualization
- `plot_trajectories()`: State trajectory plotting

**Plot Types:**
- Tracking error norm over time
- 3D state space trajectories
- Neural network weight evolution
- Comparative analysis between different agent types

## Configuration System

### Configuration Files

The system uses JSON configuration files stored in `configurations/` directory:

**Base Configuration (`config_common.json`):**
Contains shared parameters applied to all agents. Individual agent configurations override base parameters.

**Agent Configurations:**
Each JSON file defines one agent with specific parameters.

### Configuration Parameters

**Simulation Parameters:**
- `final_time` (float): Total simulation duration
- `time_step_delta` (float): Integration time step
- `num_states` (int): System state dimensionality
- `seed` (int): Random number generator seed
- `dynamics_type` (string): Dynamics model selection
- `ID` (string): Agent identifier for output files

**Neural Network Architecture:**
- `num_blocks` (int): Number of residual blocks
- `num_layers` (int): Layers per block
- `num_neurons` (int): Neurons per layer
- `output_size` (int): Network output dimensionality

**Activation Functions:**
- `inner_activation`: Hidden layer activation
- `output_activation`: Output layer activation  
- `shortcut_activation`: Residual connection activation

Available activations: "tanh", "swish", "identity", "relu", "sigmoid", "leaky_relu"

**Learning Parameters:**
- `initial_learning_rate` (float): Initial adaptation rate
- `minimum_singular_value` (float): Learning rate lower bound
- `maximum_singular_value` (float): Learning rate upper bound
- `weight_bounds` (float): Weight magnitude constraints

**Control Parameters:**
- `k1` (float): Proportional control gain

## Usage Examples

### Basic Single Agent Simulation

1. **Create configuration file** (`configurations/my_agent.json`):
```json
{
    "final_time": 60,
    "time_step_delta": 0.01,
    "num_states": 3,
    "dynamics_type": "trophic_dynamics",
    "ID": "MyAgent",
    "output_size": 3,
    "num_blocks": 1,
    "num_layers": 2,
    "num_neurons": 10,
    "inner_activation": "swish",
    "output_activation": "tanh",
    "shortcut_activation": "swish",
    "initial_learning_rate": 1.0,
    "minimum_singular_value": 0.01,
    "maximum_singular_value": 10.0,
    "weight_bounds": 5.0,
    "k1": 2.0
}
```

2. **Execute simulation**:
```bash
python main.py
```

3. **Output files generated**:
- `simulation_data/MyAgent_state_data.csv`: State trajectories
- `simulation_data/MyAgent_nn_data.csv`: Neural network data
- `simulation_data/target_state_data.csv`: Target trajectory
- Plots displayed via matplotlib

### Batch Simulation Example

**Scenario**: Compare proportional vs. neural network control

1. **Proportional controller** (`configurations/proportional.json`):
```json
{
    "ID": "Proportional",
    "k1": 1.0
}
```

2. **Neural network controller** (`configurations/neural.json`):
```json
{
    "ID": "Neural",
    "num_blocks": 2,
    "num_layers": 3,
    "num_neurons": 20,
    "k1": 1.0
}
```

3. **Common parameters** (`configurations/config_common.json`):
```json
{
    "final_time": 90,
    "time_step_delta": 0.001,
    "num_states": 3,
    "dynamics_type": "chua",
    "output_size": 3,
    "seed": 42
}
```

### Custom Dynamics Implementation

To implement custom dynamics:

1. **Modify** `src/simulation/dynamics.py`:
```python
def my_custom_dynamics(state: NDArray[np.float64]) -> NDArray[np.float64]:
    x, y, z = state
    # Your custom differential equations
    x_dot = -x + y
    y_dot = -y + z  
    z_dot = -z + x
    return np.array([x_dot, y_dot, z_dot], dtype=np.float64)
```

2. **Update dynamics map**:
```python
dynamics_map = {
    # ... existing dynamics
    "my_custom": my_custom_dynamics,
}
```

3. **Set configuration**:
```json
{
    "dynamics_type": "my_custom"
}
```

## Unit Testing

### Test Structure

The test suite covers five main areas:

**1. Batch Configuration (`test_batch_configuration.py`)**
- Tests configuration loading and merging
- Validates multi-agent setup
- Verifies baseline configuration handling

**2. End-to-End Simulation (`test_end_to_end_simulation.py`)**
- Full simulation execution test
- Regression test for tracking error performance
- Expected RMS tracking error: 8.416460028855283 (tolerance: 1e-6)

**3. Plotting Functionality (`test_plotting.py`)**
- Visualization system validation
- IEEE standard formatting verification
- Data loading and plot generation

**4. ResNet Reference (`test_resnet_reference.py`)**
- Neural network forward pass validation
- Gradient computation accuracy
- Residual connection functionality

**5. Type Hints (`test_types.py`)**
- Static type checking with mypy
- Ensures type annotation correctness
- Validates modern Python typing practices

### Test Execution

**Run all tests**:
```bash
python -m pytest tests/
```

**Run specific test module**:
```bash
python -m pytest tests/test_end_to_end_simulation.py -v
```

**Test coverage analysis**:
```bash
python -m pytest tests/ --cov=src/
```

### Testing Rationale

The test suite ensures:
- **Correctness**: Mathematical accuracy of neural network operations
- **Stability**: Consistent simulation results across runs
- **Integration**: Proper interaction between system components
- **Regression**: Protection against performance degradation
- **Type Safety**: Static analysis for runtime error prevention

## System Requirements

### Hardware Requirements

**Minimum:**
- CPU: Modern multi-core processor
- RAM: 4 GB available memory
- Storage: 1 GB free disk space

**Recommended:**
- CPU: Intel/AMD processor with AVX support
- RAM: 8 GB or higher
- Storage: SSD for faster I/O operations

### Software Dependencies

**Core Dependencies (`requirements.txt`):**
- `numpy == 2.0`: Numerical computing and array operations
- `pandas`: Data manipulation and CSV handling
- `matplotlib`: Plotting and visualization
- `scipy`: Scientific computing and numerical integration
- `SciencePlots`: IEEE standard plot formatting

**Development Dependencies (`requirements-dev.txt`):**
- `pandas-stubs`: Type hints for pandas
- `scipy-stubs`: Type hints for scipy  
- `mypy`: Static type checking
- `pytest`: Testing framework

### Performance Characteristics

**Computational Complexity:**
- Neural network forward pass: O(L × N²) where L = layers, N = neurons
- Gradient computation: O(L × N²)
- Simulation time step: O(S × N²) where S = number of states
- Total simulation: O(T × S × N²) where T = time steps

**Memory Usage:**
- Neural network weights: O(L × N²) floating point numbers
- State trajectories: O(T × S) floating point numbers
- Gradient storage: O(L × N²) floating point numbers

**I/O Performance:**
- Buffered CSV writing (100 entries per flush)
- Typical file sizes: 1-10 MB per simulation
- Plot generation: 2-5 seconds for standard visualizations

## Constraints and Limitations

### Mathematical Constraints

**Neural Network:**
- Weight bounds enforced to prevent numerical instability
- Learning rates bounded by singular value constraints
- Gradient clipping prevents exploding gradients

**Numerical Integration:**
- Fixed time step RK4 integration (no adaptive stepping)
- Limited to autonomous systems with additive control
- No built-in support for stochastic differential equations

### Computational Limitations

**Scalability:**
- Tested with up to 100 residual blocks
- Maximum recommended neurons per layer: 1000
- Time steps limited by available memory (typically < 1M steps)

**Precision:**
- Double precision floating point (64-bit)
- Numerical errors accumulate over long simulations
- Integration accuracy depends on time step size

### System Assumptions

**Dynamics Models:**
- Assumes smooth, continuously differentiable dynamics
- No support for hybrid or discrete-event systems
- Control inputs assumed to be unconstrained

**Online Learning:**
- Assumes stationary target dynamics
- No concept drift adaptation
- Learning rates may require manual tuning

**Data Management:**
- CSV format limits precision for very large/small numbers
- No built-in data compression
- Memory usage scales linearly with simulation length

### Known Issues

**Performance:**
- Large neural networks may cause memory pressure
- Long simulations generate large output files
- Real-time plotting may slow down simulation

**Compatibility:**
- Requires Python 3.10+ for modern type annotations
- SciencePlots may require LaTeX installation for full functionality
- Platform-specific differences in numerical precision possible

### Extension Points

**Adding New Dynamics:**
1. Implement dynamics function in `dynamics.py`
2. Add to dynamics map and initial conditions
3. Update configuration validation

**Custom Neural Network Architectures:**
1. Extend `NeuralNetwork` class
2. Implement custom forward/backward passes
3. Update initialization methods

**Alternative Learning Algorithms:**
1. Modify `train_step` method
2. Implement custom gradient updates
3. Add algorithm-specific parameters to configuration

This documentation provides complete technical specifications for understanding, using, and extending the Online Adaptive Deep Residual Neural Network framework.