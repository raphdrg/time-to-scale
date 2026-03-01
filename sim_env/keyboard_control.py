#!/usr/bin/env python3
"""
Keyboard control for human model in Gazebo/RViz
Controls the model's position using arrow keys or WASD
"""

import rospy
import sys
import termios
import tty
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64

class KeyboardController:
    def __init__(self):
        rospy.init_node('human_keyboard_control')
        
        # Publisher for velocity commands
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # Movement parameters
        self.linear_speed = 1.0  # m/s
        self.angular_speed = 2.0  # rad/s
        
        # Control mapping
        self.keys = {
            'w': (1, 0, 0),      # Forward
            'a': (0, 1, 0),      # Strafe left
            's': (-1, 0, 0),     # Backward
            'd': (0, -1, 0),     # Strafe right
            'q': (0, 0, 1),      # Turn left
            'e': (0, 0, -1),     # Turn right
        }
        
        self.current_vel = Twist()
        self.running = True
        
        rospy.loginfo("Keyboard Controller Started")
        rospy.loginfo("Controls:")
        rospy.loginfo("  W/A/S/D: Move forward/left/backward/right")
        rospy.loginfo("  Q/E: Turn left/right")
        rospy.loginfo("  SPACE: Stop")
        rospy.loginfo("  X: Exit")
    
    def get_key(self):
        """Get a single keypress from stdin"""
        if sys.stdin.isatty():
            settings = termios.tcgetattr(sys.stdin)
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
            return ch
        else:
            return sys.stdin.read(1)
    
    def run(self):
        """Main control loop"""
        rospy.loginfo("Ready for keyboard input...")
        
        try:
            while self.running and not rospy.is_shutdown():
                key = self.get_key().lower()
                
                if key == 'x':
                    rospy.loginfo("Exiting...")
                    break
                
                elif key == ' ':
                    # Stop
                    self.current_vel = Twist()
                    rospy.loginfo("Stopped")
                
                elif key in self.keys:
                    linear_x, linear_y, angular_z = self.keys[key]
                    
                    # Create velocity command
                    self.current_vel = Twist()
                    self.current_vel.linear.x = linear_x * self.linear_speed
                    self.current_vel.linear.y = linear_y * self.linear_speed
                    self.current_vel.angular.z = angular_z * self.angular_speed
                    
                    rospy.loginfo(f"Command: lin=({self.current_vel.linear.x:.1f}, {self.current_vel.linear.y:.1f}), ang={self.current_vel.angular.z:.1f}")
                
                # Publish the velocity command
                self.vel_pub.publish(self.current_vel)
                
        except KeyboardInterrupt:
            rospy.loginfo("Interrupted by user")
        finally:
            # Stop the model
            stop_vel = Twist()
            self.vel_pub.publish(stop_vel)
            rospy.loginfo("Stopped and exiting")

if __name__ == '__main__':
    try:
        controller = KeyboardController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
