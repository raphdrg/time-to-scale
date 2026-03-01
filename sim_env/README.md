# Human Model - Keyboard Controlled

A simple rectangular human model built with xacro URDF, with easily adjustable parameters and keyboard controls.

## Files

- **`urdf/human_model.urdf.xacro`** - Xacro URDF definition of the human as a rectangle
- **`params.yaml`** - Configuration file with adjustable model parameters
- **`keyboard_control.py`** - ROS node for keyboard control
- **`launch_human.launch`** - Launch file to run everything

## Parameters (in `params.yaml`)

Easily adjust the model by editing `params.yaml`:

```yaml
human_model:
  width: 0.4       # Width of rectangle (shoulder width) in meters
  height: 1.7      # Height of rectangle in meters
  depth: 0.2       # Depth of rectangle (front-to-back) in meters
  mass: 70.0       # Mass in kg
  color:
    r: 0.8         # Red component (0-1)
    g: 0.6         # Green component (0-1)
    b: 0.5         # Blue component (0-1)
```

## Installation & Setup

### 1. Install Gazebo Web (gzweb) for browser support

```bash
# Install gzweb
sudo apt-get install ros-noetic-gzweb
# (or on macOS: brew install gzweb)

# Or build from source
git clone https://github.com/osrf/gzweb.git
cd gzweb
./deploy.sh
```

### 2. Set up the project as a ROS package

Create a `package.xml` in the workspace root:
```bash
cat > package.xml << 'EOF'
<?xml version="1.0"?>
<package>
  <name>time_to_scale</name>
  <version>0.0.1</version>
  <description>3D SLAM with keyboard-controlled human model</description>
  <maintainer email="user@example.com">User</maintainer>
  <license>MIT</license>
  <buildtool_depend>catkin</buildtool_depend>
  <depend>rospy</depend>
  <depend>gazebo_ros</depend>
  <depend>xacro</depend>
</package>
EOF
```

## Usage

### Option 1: Launch with Gazebo (Desktop - Recommended)

```bash
# Terminal 1: Start ROS core
roscore

# Terminal 2: Launch Gazebo with the human model and keyboard control
roslaunch sim_env launch_human.launch
```

The Gazebo window will open with the human model. Use keyboard controls in Terminal 3.

```bash
# Terminal 3: Send keyboard commands (if not started by launch file)
cd /Users/raphaeldrag/Desktop/time-to-scale/sim_env
python3 keyboard_control.py
```

### Option 2: Launch Gazebo Web (Browser)

```bash
# Terminal 1: Start ROS core
roscore

# Terminal 2: Start gzserver (Gazebo headless server)
gzserver

# Terminal 3: Start gzweb (web interface server on localhost:8080)
gzweb start

# Terminal 4: Launch the human model
rosrun gazebo_ros spawn_model -urdf -model human \
  -file sim_env/urdf/human_model.urdf.xacro \
  -z 0.85

# Terminal 5: Run keyboard controller
cd sim_env && python3 keyboard_control.py
```

Then open your browser and navigate to:
```
http://localhost:8080
```

### Option 3: Desktop with RViz (Visualization Only)

```bash
# Terminal 1: Start ROS core
roscore

# Terminal 2: Launch RViz with the URDF
rosrun rviz rviz -d sim_env/urdf/human_model.rviz &
rosrun urdf_tutorial display.launch model:=sim_env/urdf/human_model.urdf.xacro

# Terminal 3: Run keyboard controller
cd sim_env && python3 keyboard_control.py
```

### Option 4: Full Setup Script (All-in-One)

Create a file `run_simulation.sh`:
```bash
#!/bin/bash
set -e

echo "Starting ROS simulation..."
export ROS_MASTER_URI=http://localhost:11311

# Start roscore in background
roscore &
ROSCORE_PID=$!
sleep 2

# Start Gazebo server
gzserver &
GZSERVER_PID=$!
sleep 1

# Start gzweb server
gzweb start &
GZWEB_PID=$!
sleep 2

# Spawn the human model
rosrun gazebo_ros spawn_model -urdf -model human \
  -file $(pwd)/sim_env/urdf/human_model.urdf.xacro \
  -z 0.85

# Run keyboard controller
cd sim_env
python3 keyboard_control.py

# Cleanup
kill $GZWEB_PID $GZSERVER_PID $ROSCORE_PID 2>/dev/null || true
```

Make it executable and run:
```bash
chmod +x run_simulation.sh
./run_simulation.sh
```

Then open: `http://localhost:8080`

## Keyboard Controls

| Key | Action |
|-----|--------|
| **W** | Move forward |
| **A** | Strafe left |
| **S** | Move backward |
| **D** | Strafe right |
| **Q** | Turn left |
| **E** | Turn right |
| **SPACE** | Stop |
| **X** | Exit |

## How It Works

1. **URDF/Xacro**: Defines the human model as a rectangular box with parametrized dimensions
2. **Inertia Calculations**: Automatically calculates moment of inertia based on dimensions and mass
3. **Keyboard Input**: The Python script reads keyboard input and publishes velocity commands to `/cmd_vel`
4. **Gazebo**: Applies the velocity commands to move the model in simulation

## Customization

To change the model dimensions and appearance, edit `params.yaml` and either:
- Re-run the launch file, or
- Pass parameters directly when spawning:
  ```bash
  rosrun gazebo_ros spawn_model -urdf -model human \
    -file sim_env/urdf/human_model.urdf.xacro \
    -param human_model width:=0.5 height:=1.8 -z 0.85
  ```

## Troubleshooting

**gzweb not starting?**
```bash
# Kill existing processes
killall gzserver gzweb roscore 2>/dev/null

# Start fresh
roscore
gzserver
gzweb start
```

**Port 8080 already in use?**
```bash
# Find and kill process on port 8080
lsof -ti:8080 | xargs kill -9

# Or use a different port (edit gzweb config)
```

**Model not spawning?**
- Check the URDF file path is correct
- Verify gazebo is running: `ps aux | grep gazebo`
- Check ROS_MASTER_URI is set correctly

**Keyboard control not working?**
- Ensure keyboard controller terminal has focus
- Run from the correct directory with the script
- Check /cmd_vel topic: `rostopic list | grep cmd_vel`
