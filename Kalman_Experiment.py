import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.stats import norm
from filterpy.kalman import KalmanFilter


######################################    First is the direct Kalman filtering with observtion in x only    ##########################################################################
# Define time parameters
dt = 0.1  # Time step
total_time = 5.0  # Total time
time_steps = int(total_time / dt)
time_array = np.linspace(0, total_time, time_steps)  # Time array

# Define the state transition matrix (for continuous time)
A = np.array([[0, 1], [0, 0]])  # Second-order system, the first derivative of x is dx/dt
F = expm(A * dt)  # State transition matrix for discrete time

# Define the observation matrix
H = np.array([[1, 0]])  # Only observe position x

# Covariance matrices for process noise and observation noise
Q = np.eye(2) * 0.01
R = np.array([[1]]) * 0.01

# Create a Kalman Filter using the filterpy library
kf = KalmanFilter(dim_x=2, dim_z=1)
kf.F = F
kf.H = H
kf.Q = Q
kf.R = R
kf.P = np.eye(2)  # Initial state covariance
kf.x = np.array([[0], [0]])  # Initial state

# Store the results
X_estimated_filterpy = np.zeros((time_steps, 2))
X_manual_diff = np.zeros(time_steps - 1)

# Process simulation and Kalman filtering
for k in range(time_steps):
    # Prediction
    kf.predict()

    # Generate simulated observation
    Y = H @ kf.x + norm.rvs(0, np.sqrt(R), size=(1, 1))

    # Update
    kf.update(Y)

    # Store estimated values
    X_estimated_filterpy[k, :] = kf.x.T

    # Calculate manual differentiation (only when k > 0)
    if k > 0:
        X_manual_diff[k - 1] = (X_estimated_filterpy[k, 0] - X_estimated_filterpy[k - 1, 0]) / dt

# Plotting
plt.figure(figsize=(12, 6))

# Plot position x
plt.subplot(2, 1, 1)
plt.plot(time_array, X_estimated_filterpy[:, 0], label="FilterPy Estimated Position (x)")
plt.title("Position (x) and Velocity (dx/dt) - FilterPy Kalman Filter")
plt.ylabel("Position (x)")
plt.legend()

# Plot velocity dx/dt
plt.subplot(2, 1, 2)
plt.plot(time_array[:-1], X_manual_diff, label="Manually Differentiated Velocity (dx/dt)")
plt.plot(time_array, X_estimated_filterpy[:, 1], label="FilterPy Estimated Velocity (dx/dt)", alpha=0.7)
plt.xlabel("Time")
plt.ylabel("Velocity (dx/dt)")
plt.legend()

plt.tight_layout()
plt.show()






#################################################  Second is Kalman filtering with dx/dt observation formed by differentiating x observation #######################################

# Define time parameters
dt = 0.1  # Time step
total_time = 5.0  # Total time
time_steps = int(total_time / dt)
time_array = np.linspace(0, total_time, time_steps)  # Time array

# Define the state transition matrix (for continuous time)
A = np.array([[0, 1], [0, 0]])  # Second-order system
F = expm(A * dt)  # State transition matrix for discrete time

# Define the observation matrix - Identity matrix for observing both x and dx/dt
H = np.eye(2)

# Covariance matrices for process noise and observation noise
Q = np.eye(2) * 0.01
R = np.eye(2) * 0.01  # Adjust this if necessary

# Create a Kalman Filter using the filterpy library
kf = KalmanFilter(dim_x=2, dim_z=2)
kf.F = F
kf.H = H
kf.Q = Q
kf.R = R
kf.P = np.eye(2)  # Initial state covariance
kf.x = np.array([[0], [0]])  # Initial state

# Store the results
X_estimated_filterpy = np.zeros((time_steps, 2))
observed_velocity = np.zeros(time_steps - 1)

# Generate simulated observation for position x
observed_position = norm.rvs(size=time_steps) * np.sqrt(R[0, 0]) + np.cumsum(np.ones(time_steps) * dt)

# Process simulation and Kalman filtering
for k in range(1, time_steps):
    # Differentiate to estimate velocity dx/dt
    observed_velocity[k - 1] = (observed_position[k] - observed_position[k - 1]) / dt

    # Form the observation vector
    Y = np.array([[observed_position[k]], [observed_velocity[k - 1]]])

    # Prediction
    kf.predict()

    # Update using the observation
    kf.update(Y)

    # Store estimated values
    X_estimated_filterpy[k, :] = kf.x.T

# Plotting
plt.figure(figsize=(12, 6))

# Plot position x
plt.subplot(2, 1, 1)
plt.plot(time_array, observed_position, label="Observed Position (x)")
plt.plot(time_array, X_estimated_filterpy[:, 0], label="FilterPy Estimated Position (x)", alpha=0.7)
plt.title("Position (x) and Velocity (dx/dt) - FilterPy Kalman Filter")
plt.ylabel("Position (x)")
plt.legend()

# Plot velocity dx/dt
plt.subplot(2, 1, 2)
plt.plot(time_array[1:], observed_velocity, label="Observed Velocity (dx/dt)")
plt.plot(time_array, X_estimated_filterpy[:, 1], label="FilterPy Estimated Velocity (dx/dt)", alpha=0.7)
plt.xlabel("Time")
plt.ylabel("Velocity (dx/dt)")
plt.legend()

plt.tight_layout()
plt.show()
