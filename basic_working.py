from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

def DE(t, y, L):
    a_w, a_b = y
    a_g = p - a_w - a_b

    temps = calc_temps(y, L)

    beta_w = beta_T(temps[1])
    beta_b = beta_T(temps[2])

    DE1 = a_w * (a_g * beta_w - gamma)
    DE2 = a_b * (a_g * beta_b - gamma)
    return [DE1, DE2]

def beta_T(T):
    return 1 - k * (T - T_opt) ** 2 if abs(T - T_opt) < (1 / np.sqrt(k)) else 0
     
def calc_temps(y, L):
    a_w, a_b = y
    a_g = p - a_w - a_b

    alpha_p = a_w * alpha_w + a_b * alpha_b + a_g * alpha_g

    T4 = s_0 * L * (1 - alpha_p) / sigma

    T_w4 = q * (alpha_p - alpha_w) + T4
    T_b4 = q * (alpha_p - alpha_b) + T4
    T_g4 = q * (alpha_p - alpha_g) + T4

    return [T4**0.25, T_w4**0.25, T_b4**0.25, T_g4**0.25]

# Declaring Variables
sigma = 5.67e-8  # Boltzmann constant
s_0 = 917  # Solar irradiance
q = 2.06e9  # Heat transfer coefficient
T_1 = 278  # Minimum temperature
T_2 = 313  # Maximum temperature
T_opt = (T_1 + T_2) / 2  # Optimal Temperature
k = 1 / ((T_2 - T_1) / 2) ** 2  # Parabolic width
alpha_g = 0.5  # Albedo of the ground
alpha_w = 0.75  # Albedo of white daisies
alpha_b = 0.25  # Albedo of black daisies  
p = 1  # Fraction of land that is fertile
gamma = 0.3  # Death rate

# Initial conditions
a_w0 = 0.01  # Initial fraction of white daisies
a_b0 = 0.01  # Initial fraction of black daisies
L = 1.0  # Fixed luminosity 
t_span = (0, 1000)

# Solve the system
initial = [a_w0, a_b0]
res = solve_ivp(DE, t_span, initial, args=(L,), dense_output=True, rtol=1e-8, atol=1e-8)

# Create time points for plotting
t = np.linspace(0, 50, 1000)
solution = res.sol(t)

# Plot population fractions
plt.figure(figsize=(10, 6))
plt.plot(t, solution[0], 'b-', label='White Daisies')
plt.plot(t, solution[1], 'k-', label='Black Daisies')
plt.plot(t, p - solution[0] - solution[1], 'g-', label='Bare Ground')
plt.xlabel('Time')
plt.ylabel('Area Fraction')
plt.legend()
plt.grid(True)
plt.show()

# Calculate and plot temperatures
temps_array = np.array([calc_temps([solution[0][i], solution[1][i]], L) for i in range(len(t))])

plt.figure(figsize=(10, 6))
plt.plot(t, temps_array[:, 0] - 273.15, 'r-', label='Planetary Temperature')
plt.plot(t, temps_array[:, 1] - 273.15, 'b--', label='White Daisy Temperature')
plt.plot(t, temps_array[:, 2] - 273.15, 'k--', label='Black Daisy Temperature')
plt.axhline(y=T_opt - 273.15, color='y', linestyle=':', label='Optimal Temperature')
plt.xlabel('Time')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.show()
