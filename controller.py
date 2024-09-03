# controller.py

import numpy as np


class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def update(self, error, dt):
        """ Update the PID control output """
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output


class Autopilot:
    def __init__(self):
        # Increased PID Controllers gains for x, y, z axes
        self.x_controller = PIDController(2.5, 0.0, 1)  # Increase kp
        self.y_controller = PIDController(2.5, 0.0, 1)  # Increase kp
        self.z_controller = PIDController(3.0, 0.0, 1.0)  # Increase kp and kd

        # Gravity compensation term
        self.gravity_compensation = 9.81  # to counteract gravity on z-axis

    def compute_control(self, target_pos, current_pos, dt):
        """ Compute the forces to apply to the drone to move towards the target """
        error = target_pos - current_pos[:3]  # position error

        # Compute control outputs
        fx = self.x_controller.update(error[0], dt)
        fy = self.y_controller.update(error[1], dt)
        fz = self.z_controller.update(error[2], dt) + self.gravity_compensation

        # The forces are computed assuming simple proportional control
        return np.array([fx, fy, fz])
