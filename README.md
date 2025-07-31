# Online Adaptive Deep Residual Neural Network

An implementation of online adaptive deep neural networks with residual connections for modeling and controlling dynamical systems. The system simulates autonomous dynamical systems and continuously adapts neural network weights in real time.

## Quickstart Guide

### Prerequisites

- Python 3.11 or higher
- Dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:
```bash
git clone https://github.com/cristian1928/Online-Adaptive-Deep-Residual-Neural-Network.git
cd Online-Adaptive-Deep-Residual-Neural-Network
```

2. Create and activate a virtual environment:
```bash
# Create the environment
python3 -m venv venv

# Activate on macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt -r requirements-dev.txt
```

### How to Run the Program

Execute the main simulation:
```bash
python3 main.py
```

This command:
- Loads configuration files from the `configurations/` directory
- Creates agents based on configuration parameters
- Runs the simulation with online adaptive learning
- Generates output data in the `simulation_data/` directory
- Creates IEEE-standard plots for result analysis

### Viewing the Results

After the simulation completes and generates data in the `simulation_data/` directory, you can create and display all result plots by running the plotting script:

```bash
python3 src/visualization/plotter.py
```

## High-Level Overview

### Purpose

This framework implements online adaptive control using deep residual neural networks for nonlinear dynamical systems. The system performs real-time target tracking where neural networks continuously adapt their weights based on tracking error feedback.

### Main Components

- **Neural Network Core**: Deep residual architecture with online weight adaptation
- **Dynamical Systems**: Built-in models including attitude control, ecological dynamics, and chaotic systems
- **Simulation Engine**: Numerical integration and real-time state evolution
- **Data Management**: Efficient CSV logging with buffered I/O
- **Visualization**: IEEE-standard plotting and performance analysis

### Supported Dynamics

- **Attitude MRP**: Spacecraft attitude control using Modified Rodrigues Parameters
- **Chua Circuit**: Chaotic double-scroll circuit dynamics
- **Trophic Dynamics**: Three-tier ecological food chain model
- **Custom**: User-defined dynamical systems

### Configuration

The system uses JSON configuration files in the `configurations/` directory. Each file defines simulation parameters, neural network architecture, and learning parameters for one agent. Multiple configuration files enable batch simulations with different agents.

## Code Quality & Testing

Before committing or pushing changes, it's recommended to run the following checks to ensure code quality and correctness.

### 1. Static Type Checking

Run this after making any changes to function signatures or data types.

```bash
mypy --strict 
```

### 2. Testing

This runs the full suite of unit and integration tests.

```bash
python -m pytest -v tests/
```

## License

GNU Affero General Public License v3.0. See [LICENSE](LICENSE) for details.

## Citation

```
@article{Nino.Patil.ea2025,
  author        = {Cristian F. Nino and Omkar Sudhir Patil and Marla R. Eisman and Warren E. Dixon},
  title         = {Online ResNet-Based Adaptive Control for Nonlinear Target Tracking},
  year          = {2025},
  journal       = {IEEE Control Systems Letters},
  volume        = {9},
  pages         = {907-912},
  doi           = {10.1109/LCSYS.2025.3576652}
}
```

## Contact

For questions, suggestions, or further information, please contact:

- **Name:** Cristian Nino
- **Email:** cristian1928@ufl.edu
- **GitHub:** [@cristian1928](https://github.com/cristian1928)

For detailed technical documentation, see [HELP.md](HELP.md).