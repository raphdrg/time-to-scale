#!/bin/bash
set -e

IMAGE_NAME="time-to-scale:ros2-jazzy"
PROJECT_DIR="/Users/raphaeldrag/Desktop/time-to-scale"

echo "🚀 Building ROS 2 Jazzy Docker image..."
echo "This builds a minimal ROS 2 Jazzy environment with xacro and ros_gz."
echo ""

docker rmi "$IMAGE_NAME" 2>/dev/null || true
docker build -t "$IMAGE_NAME" .

echo ""
echo "✓ Image built! Starting container..."
echo ""

docker run -it --rm \
  --platform linux/arm64 \
  -v "$PROJECT_DIR":/workspace \
  "$IMAGE_NAME" bash -c "
    source /opt/ros/jazzy/setup.bash
    echo '✓ ROS 2 Jazzy environment ready!'
    echo ''
    echo 'Available in this container:'
    echo '  - ROS 2 Jazzy'
    echo '  - xacro'
    echo '  - ros_gz / Gazebo integration packages'
    echo '  - Your project mounted at /workspace'
    echo ''
    echo 'Quick commands:'
    echo '  cd /workspace'
    echo '  xacro /workspace/sim_env/urdf/human_model.urdf.xacro'
    echo '  ros2 pkg list | grep xacro'
    echo '  ros2 pkg list | grep ros_gz'
    echo ''
    echo 'Note: Gazebo GUI on macOS Docker is often limited.'
    echo ''
    cd /workspace
    exec bash
  "
  