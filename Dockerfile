FROM ros:jazzy

ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-xacro \
    ros-jazzy-ros-gz \
    python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc

CMD ["bash"]