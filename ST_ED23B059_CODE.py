import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Constants
m = 260  # kg
p = 1.225  # kg/L
A = 1  # m^2
Cd = 0.12
crr1 = 0.012
g = 9.81  # m/s^2
T = 20  # Time interval (s)
Qmax = 1000  # Maximum capacity of the battery (Wh)

# Time, Velocity, and Acceleration
t = tf.constant(np.linspace(0, 20, 10), dtype=tf.float32)
vel_1 = np.array([81, 90, 65, 34, 43, 0, 25, 25, 90, 2], dtype=np.float32)
vel = tf.Variable(vel_1, dtype=tf.float32, name='vel')
acc = tf.Variable(tf.zeros_like(vel), dtype=tf.float32, name='acc')

# Compute acceleration
for i in range(vel.shape[0]-1):
    acc[i].assign((vel[i+1]-vel[i])/(t[i+1]-t[i]))

# Solar panel output function
def solar_panel_output(t):
    a = tf.constant(0.00882316, dtype=tf.float32)
    b = tf.constant(-0.68036, dtype=tf.float32)
    c = tf.constant(21.4325, dtype=tf.float32)
    d = tf.constant(-352.414, dtype=tf.float32)
    e = tf.constant(3159.69, dtype=tf.float32)
    f = tf.constant(-14330.4, dtype=tf.float32)
    g = tf.constant(25444.8, dtype=tf.float32)
    return a*t**3 - b*t**5 + c*t**4 - d*t**3 + e*t**2 - f*t + g

# Objective function
def obj_fun(vel):
    vel_diff = vel[1:] - vel[:-1]
    return solar_panel_output(t) - vel * (m * acc + 0.5 * p * A * Cd * tf.reduce_sum(tf.square(vel_diff)) + m * g * crr1)

# Adam optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.03)

# Lists to store values for plotting
initial_values = []
optimized_values = []

# Optimization loop
tolerance = 1e-5
prev_z = None
for _ in range(100):
    with tf.GradientTape() as tape:
        z = obj_fun(vel)
    gradients = tape.gradient(z, [vel])
    optimizer.apply_gradients(zip(gradients, [vel]))
    if prev_z is not None:
        if np.abs(prev_z - z.numpy()[0]) < tolerance:
            break
    initial_values.append(tf.reduce_sum(obj_fun(vel)).numpy())
    prev_z = z.numpy()[0]

# Print initial and final optimized velocity values
print("Initial Velocity Values:")
print(vel_1)
print('\nFinal Optimized Velocity Values:')
print(vel.numpy())

# Calculate State of Charge (SoC)for initial and optimized obj_fun values
initial_integral_obj_fun = np.trapz(obj_fun(vel_1).numpy(), dx=T)
final_integral_obj_fun = np.trapz(obj_fun(vel.numpy()).numpy(), dx=T)
initial_soc = (100 * T * initial_integral_obj_fun) / Qmax
final_soc = (100 * T * final_integral_obj_fun) / Qmax

# Print State of Charge (SoC) for initial and final optimized obj_fun values
print("\nInitial SoC:", initial_soc)
print("Final SoC:", final_soc)

# Plot solar power output
plt.figure(figsize=(10, 6))
plt.plot(t, solar_panel_output(t), label='Solar Output Power')
plt.xlabel('Time (s)')
plt.ylabel('Power (W)')
plt.title('Solar Panel Output Power')
plt.legend()
plt.grid(True)
plt.show()

# Plot initial and optimized velocity profiles vs. time
plt.figure(figsize=(10, 6))
plt.plot(t, vel_1, label='Initial Velocity Profile')
plt.plot(t, vel.numpy(), label='Optimized Velocity Profile', linestyle='--')
plt.xlabel('Time(s)')
plt.ylabel('Velocity (m/s)')
plt.title('Initial and Optimized Velocity Profiles')
plt.legend()
plt.grid(True)
plt.show()

# Plot initial and optimized objective function values vs. time
plt.figure(figsize=(10, 6))
plt.plot(t, obj_fun(tf.constant(vel_1, dtype=tf.float32)).numpy(),
          label='Initial obj_fun')
plt.plot(t, obj_fun(tf.constant(vel.numpy(), dtype=tf.float32)).numpy(),
          label='Final Optimized obj_fun', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Objective Function Value')
plt.title('Initial and Optimized Objective Function Values vs. Time')
plt.legend()
plt.grid(True)
plt.show()

# Plot initial obj_fun vs. initial velocity profile and final optimized obj_fun vs. final optimized velocity profile
plt.figure(figsize=(10, 6))
plt.plot(vel_1, obj_fun(tf.constant(vel_1, dtype=tf.float32)).numpy(),
          label='Initial obj_fun vs. Initial Velocity Profile')
plt.plot(vel.numpy(), obj_fun(tf.constant(vel.numpy(), dtype=tf.float32)).numpy(),
          label='Final Optimized obj_fun vs. Final Optimized Velocity Profile',
          linestyle='--')
plt.xlabel('Velocity (m/s)')
plt.ylabel('Objective Function Value')
plt.title('Objective Function Values vs. Velocity Profiles')
plt.legend()
plt.grid(True)
plt.show()