# time-to-scale

3D SLAM with an iPhone. Reconstructing environments with fidelity.


## Project Structure

```
time-to-scale/
├── README.md                          # This file
├── Dockerfile                         # ARM64-optimized ROS 2 Jazzy image
├── run_docker.sh                      # macOS: Quick docker launcher
├── sim_env/                           # Simulation environment
│   ├── README.md                      # Detailed sim documentation
│   ├── params.yaml                    # Model parameters (adjustable)
│   ├── keyboard_control.py            # Keyboard controller script
│   ├── launch_human.launch            # ROS launch file
│   ├── worlds/
│   │   └── barriers.world             # Gazebo world with 45° barriers
│   └── urdf/
│       └── human_model.urdf.xacro     # URDF human model definition
```

## World Description

The **barriers.world** Gazebo environment features:
- **Ground plane** - Flat 100x100m surface
- **6 barrier obstacles** - 3m wide × 1m tall rectangular barriers positioned at 45-degree angles
- **Varied orientations** - Alternating 45° angles to create an obstacle course for testing navigation

Perfect for testing collision detection and navigation algorithms with the keyboard-controlled human model.

## Project Dependencies

### Core Requirements

| Package | Version | Purpose | 
|---------|---------|---------|
| **ROS** | Noetic (Ubuntu) / Latest (macOS) | Robot Operating System framework |
| **Gazebo** | 11.0+ | Physics simulation engine |
| **xacro** | Latest | URDF macro language processor |
| **Python** | 3.6+ | Keyboard controller scripting |
| **gazebo_ros** | Latest | ROS/Gazebo integration packages |
| **gazebo_plugins** | Latest | Physics and sensor plugins |

### Python Dependencies

The keyboard controller uses only Python standard library (no external packages required):
- `sys` - System interface
- `termios` - Unix terminal control
- `tty` - Terminal control
- `rospy` - ROS Python client (installed with ROS)

## Quick Setup (Complete Commands)

### macOS (Apple Silicon)

```bash
# 1. Ensure you're in the project directory
cd /Users/raphaeldrag/Desktop/time-to-scale

# 2. Run the launcher (builds ARM64 image on first run)
./run_docker.sh

# Inside the container:
gazebo /workspace/sim_env/worlds/barriers.world
```

**That's it!** The custom Dockerfile handles everything:
- ✅ Builds a custom ARM64-optimized ROS 2 image (2-5 min first time)
- ✅ Installs Gazebo, xacro, and ROS packages
- ✅ Mounts your project
- ✅ Auto-sources ROS 2

### Linux

```bash
# Install ROS 2 and dependencies
sudo curl -sSL https://repo.ros2.org/ros.key | sudo apt-key add -
sudo sh -c 'echo "deb [arch=amd64] http://packages.ros2.org/ubuntu jammy main" > /etc/apt/sources.list.d/ros2.list'
sudo apt update
sudo apt install -y ros-jazzy-desktop gazebo xacro python3-rosdep

# Source ROS
source /opt/ros/jazzy/setup.bash

# Run the simulation
cd /Users/raphaeldrag/Desktop/time-to-scale
gazebo sim_env/worlds/barriers.world
```

## Prerequisites & Installation

This project requires ROS and Gazebo simulation environment. **On macOS, we use Docker** since native ROS support is limited.

### macOS (Recommended: Use Docker)

**For Apple Silicon Macs (M1/M2/M3)**, we provide a custom Dockerfile optimized for ARM64:

```bash
# 1. Navigate to the project
cd /Users/raphaeldrag/Desktop/time-to-scale

# 2. Run the launcher script (it builds the image automatically)
./run_docker.sh

# First run takes 2-5 minutes to build the image
# Subsequent runs will be instant
```

The script will:
- ✅ Build a custom ARM64-optimized ROS 2 Jazzy Docker image
- ✅ Mount your project at `/workspace`
- ✅ Install Gazebo and all dependencies
- ✅ Auto-source ROS 2 setup

**You're now inside a Linux container with ROS 2!**

```bash
# Inside the container, test the installation
gazebo --version
xacro --version
ros2 --version

# Launch Gazebo with barriers
gazebo /workspace/sim_env/worlds/barriers.world
```

### Linux (Ubuntu/Debian)

```bash
# Add ROS 2 repository
sudo curl -sSL https://repo.ros2.org/ros.key | sudo apt-key add -
sudo sh -c 'echo "deb [arch=amd64] http://packages.ros2.org/ubuntu jammy main" > /etc/apt/sources.list.d/ros2.list'

# Update and install
sudo apt update
sudo apt install -y ros-jazzy-desktop gazebo ros-jazzy-gazebo-ros ros-jazzy-gazebo-plugins xacro python3

# Source ROS (add to ~/.bashrc for automatic sourcing)
source /opt/ros/jazzy/setup.bash
```

### Quick Dependency Check

```bash
# Check all required tools are installed
echo "=== Dependency Check ==="
echo "ROS version:"
roscore --version 2>/dev/null || echo "❌ ROS not found"

echo "Gazebo version:"
gazebo --version 2>/dev/null || echo "❌ Gazebo not found"

echo "xacro version:"
xacro --version 2>/dev/null || echo "❌ xacro not found"

echo "Python version:"
python3 --version 2>/dev/null || echo "❌ Python3 not found"
```

## Quick Start: Running the Simulation

### macOS (Using Docker with ARM64 Support)

**Exit the current container first:**

```bash
exit
```

**Then run the proper setup:**

```bash
cd /Users/raphaeldrag/Desktop/time-to-scale
./run_docker.sh
```

This will build the custom Docker image (takes 2-5 min first time) and drop you into a container with ROS 2 Jazzy ready.

**Inside the container:**

```bash
# Launch Gazebo with barriers
gazebo /workspace/sim_env/worlds/barriers.world &

# In another terminal window, spawn the human model
docker exec -it <container-id> bash
cd /workspace
ros2 run gazebo_ros spawn_model -entity human -file /workspace/sim_env/urdf/human_model.urdf.xacro
```

**Find container ID:**
```bash
docker ps  # Shows running containers
```

### Linux

**Terminal 1 - Source ROS and launch Gazebo:**

```bash
source /opt/ros/jazzy/setup.bash
gazebo /path/to/sim_env/worlds/barriers.world
```

**Terminal 2 - Spawn the model:**

```bash
source /opt/ros/jazzy/setup.bash
ros2 run gazebo_ros spawn_model -entity human -file /path/to/sim_env/urdf/human_model.urdf.xacro
```

**Terminal 3 - Run keyboard controller (optional):**

```bash
cd /path/to/sim_env
python3 keyboard_control.py
```

Controls: **W/A/S/D** to move, **Q/E** to rotate, **SPACE** to stop, **X** to exit

## Full Installation Checklist

- [ ] Installed ROS (version: __________)
- [ ] Installed Gazebo (version: __________)
- [ ] Installed xacro
- [ ] Installed Python 3.6+
- [ ] Sourced ROS setup script in shell config
- [ ] Verified all tools with `roscore --version`, `gazebo --version`, `xacro --version`
- [ ] (Optional) Installed gzweb for browser access
- [ ] Downloaded/cloned this project
- [ ] Tested with `roslaunch sim_env launch_human.launch`

## Troubleshooting

### Installation Issues

**"ROS command not found"**
- Make sure you sourced the ROS setup script: `source /opt/ros/noetic/setup.bash` (Linux) or `source /opt/homebrew/opt/ros/setup.zsh` (macOS)
- Add it to your shell config file so it runs automatically

**"Gazebo command not found"**
```bash
# macOS
brew install gazebo

# Linux
sudo apt install gazebo ros-noetic-gazebo-ros
```

**"xacro command not found"**
```bash
# macOS
brew install xacro

# Linux
sudo apt install ros-noetic-xacro
```

### Runtime Issues

**"gzserver: command not found"**
```bash
# Install gazebo-ros integration
sudo apt install ros-noetic-gazebo-ros

# Or on macOS
brew install gazebo ros-noetic-gazebo-ros
```

**Port 8080 already in use (gzweb)**
```bash
# Kill existing process
lsof -ti:8080 | xargs kill -9

# Or find and kill gzweb specifically
pkill -f gzweb
```

**Model not appearing in simulation**
- Check Gazebo is running: `ps aux | grep gazebo`
- Try spawning manually: `rosrun gazebo_ros spawn_model -urdf -model test -file sim_env/urdf/human_model.urdf.xacro -z 0.85`
- Check file paths are absolute or run from correct directory

**Keyboard controller not responding**
- Ensure terminal has focus (click on it)
- Verify `/cmd_vel` topic exists: `rostopic list | grep cmd_vel`
- Check ROS is running: `rostopic list` should show topics
- Run controller with: `python3 keyboard_control.py` from `sim_env` directory

## Next Steps

- See [sim_env/README.md](sim_env/README.md) for detailed simulation documentation
- Modify `sim_env/params.yaml` to adjust the human model dimensions
- Add more complex URDF models to the `sim_env/urdf/` directory
- Integrate iPhone SLAM data with the simulation
