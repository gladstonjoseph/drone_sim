# main.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
from quadcopter import Quadcopter
from controller import Autopilot
import utils


def run_simulation(simulation_mode='real-time'):
    # Initialize the quadcopter and autopilot
    quad = Quadcopter()
    autopilot = Autopilot()

    # Set initial conditions
    dt = 0.01  # time step
    target_position = np.array([5.0, 10.0, 5.0])  # Target position (x, y, z)

    # Data storage for plotting and exporting
    time_data = []
    position_data = []
    state_data = []

    # Run the simulation
    for i in range(1000):  # simulate for 1000 time steps
        current_state = quad.get_state()
        current_position = current_state[:3]  # x, y, z

        # Store data for plotting and exporting
        time_data.append(i * dt)
        position_data.append(current_position)
        state_data.append(current_state)

        # Compute the control forces
        forces = autopilot.compute_control(target_position, current_state, dt)

        # Convert control forces to body frame forces
        rotation = utils.rotation_matrix(*current_state[6:9])
        body_forces = rotation.T @ forces

        # Set forces and moments in the quadcopter model
        quad.set_forces(body_forces)
        quad.set_moments(np.zeros(3))  # Assuming no rotational control for simplicity

        # Update the quadcopter state
        quad.dynamics(dt)

    # Convert position data and state data to numpy array for easier manipulation
    position_data = np.array(position_data)

    # Determine dynamic limits for the plot
    x_min, y_min, z_min = position_data.min(axis=0)
    x_max, y_max, z_max = position_data.max(axis=0)

    # Export the data to a CSV file including yaw angle
    columns = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'roll', 'pitch', 'yaw', 'p', 'q', 'r']
    df = pd.DataFrame(state_data, columns=columns)
    df.insert(0, 'time', time_data)  # Add time column at the beginning
    df.to_csv('drone_simulation_data.csv', index=False)
    print("Data exported to drone_simulation_data.csv")

    # Visualization
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Set dynamic limits based on actual data ranges
    ax.set_xlim([x_min - 1, x_max + 1])
    ax.set_ylim([y_min - 1, y_max + 1])
    ax.set_zlim([z_min - 1, z_max + 1])

    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title('Trajectory in 3D Space')

    # Mark the starting and target positions
    ax.scatter(0, 0, 0, color='green', marker='o', s=100, label='Start Position')
    ax.scatter(target_position[0], target_position[1], target_position[2], color='red', marker='x', s=100, label='Target Position')

    line, = ax.plot([], [], [], lw=2, label='Drone Path')
    ax.legend()

    if simulation_mode == 'real-time':
        # Real-time simulation
        def update(frame):
            line.set_data(position_data[:frame, 0], position_data[:frame, 1])
            line.set_3d_properties(position_data[:frame, 2])
            return line,

        ani = FuncAnimation(fig, update, frames=len(position_data), interval=10, blit=True)
        plt.show()

    elif simulation_mode == 'quick':
        # Quick simulation, plot final result
        line.set_data(position_data[:, 0], position_data[:, 1])
        line.set_3d_properties(position_data[:, 2])
        plt.show()


if __name__ == "__main__":
    # mode = input("Enter simulation mode ('real-time' or 'quick'): ").strip().lower()
    # if mode not in ['real-time', 'quick']:
    #     print("Invalid mode. Defaulting to 'quick'.")
    #     mode = 'quick'
    run_simulation(simulation_mode='quick')
