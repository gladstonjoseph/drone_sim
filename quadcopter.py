# quadcopter.py

import numpy as np


class Quadcopter:
    def __init__(self, mass=1.0, arm_length=0.25, inertia=np.array([0.01, 0.01, 0.02])):
        self.mass = mass
        self.arm_length = arm_length
        self.inertia = np.diag(inertia)  # Corrected to be a 3x3 diagonal matrix

        # State: [x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r]
        self.state = np.zeros(12)

        # Forces and moments
        self.forces = np.zeros(3)
        self.moments = np.zeros(3)

        # Gravity
        self.gravity = np.array([0, 0, -9.81 * mass])

    def set_forces(self, forces):
        """ Set the total forces acting on the quadcopter """
        self.forces = forces

    def set_moments(self, moments):
        """ Set the total moments acting on the quadcopter """
        self.moments = moments

    def dynamics(self, dt):
        """ Compute the next state based on the current state and inputs """
        x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r = self.state

        # Translational dynamics
        acc = (self.forces + self.gravity) / self.mass
        vx += acc[0] * dt
        vy += acc[1] * dt
        vz += acc[2] * dt
        x += vx * dt
        y += vy * dt
        z += vz * dt

        # Rotational dynamics
        angular_acc = np.linalg.inv(self.inertia) @ (self.moments - np.cross([p, q, r], self.inertia @ [p, q, r]))
        p += angular_acc[0] * dt
        q += angular_acc[1] * dt
        r += angular_acc[2] * dt

        # Update roll, pitch, yaw
        roll += p * dt
        pitch += q * dt
        yaw += r * dt

        self.state = np.array([x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r])

    def get_state(self):
        """ Return the current state """
        return self.state
