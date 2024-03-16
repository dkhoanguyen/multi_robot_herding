import numpy as np
import matplotlib.pyplot as plt

# Define double integrator dynamics
def double_integrator_dynamics(x, u):
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])
    return A.dot(x) + B.dot(u)

# Define Control Barrier Function (CBF)
def control_barrier_function(x):
    # Safety constraint: x1 + x2 >= 1
    return x[0] + x[1] - 1

# Define CBF derivative w.r.t. state (x) and control input (u)
def cbf_derivative_x(x):
    return np.array([1, 1])

def cbf_derivative_u():
    return 0

# Controller using Barrier Function
def controller(x):
    # Define control law using Barrier Function
    k = 1.0  # Gain
    u = -k * cbf_derivative_x(x) / cbf_derivative_u()
    return u

# Simulation parameters
dt = 0.01  # Time step
T = 5.0    # Total simulation time
num_steps = int(T / dt)

# Initial state
x0 = np.array([0.5, 0.5])

# Lists to store states and control inputs for plotting
x_hist = [x0]
u_hist = []

# Simulation loop
x = x0
for t in range(num_steps):
    # Get control input from controller
    u = controller(x)
    
    # Apply control input and simulate dynamics
    x = x + dt * double_integrator_dynamics(x, u)
    
    # Store state and control input
    x_hist.append(x)
    u_hist.append(u)

# Convert lists to arrays for easier manipulation
x_hist = np.array(x_hist)
u_hist = np.array(u_hist)

# Plotting
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(np.arange(0, T + dt, dt), x_hist[:, 0], label='x1')
plt.plot(np.arange(0, T + dt, dt), x_hist[:, 1], label='x2')
plt.xlabel('Time')
plt.ylabel('State')
plt.title('Double Integrator System States')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(np.arange(0, T, dt), u_hist[:, 0], label='u')
plt.xlabel('Time')
plt.ylabel('Control Input')
plt.title('Control Input')
plt.legend()

plt.tight_layout()
plt.show()
 