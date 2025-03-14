# Importing libraries
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from matplotlib.gridspec import GridSpec

# Function defining the system of differential equations
def DE(t, y, L):
    a_w, a_b = y
    a_g = p - a_w - a_b

    # Calculate local temperatures
    temps = calc_temps(y, L)
    
    # Growth rates based on local temperatures
    beta_w = beta_T(temps[1])
    beta_b = beta_T(temps[2])
    
    # Differential equations describing the change in daisy populations
    DE1 = a_w * (a_g * beta_w - gamma)
    DE2 = a_b * (a_g * beta_b - gamma)

    return [DE1, DE2]

# Function to calculate growth rate based on temperature
def beta_T(T):
    return 1 - k * (T - T_opt) ** 2 if abs(T - T_opt) < (1 / np.sqrt(k)) else 0

# Function to calculate temperatures of planet and daisy types
def calc_temps(y, L):
    a_w, a_b = y
    a_g = p - a_w - a_b

    alpha_p = a_w * alpha_w + a_b * alpha_b + a_g * alpha_g

    T4 = s_0 * L * (1 - alpha_p) / sigma

    T_w4 = q * (alpha_p - alpha_w) + T4
    T_b4 = q * (alpha_p - alpha_b) + T4
    T_g4 = q * (alpha_p - alpha_g) + T4

    return [T4**0.25, T_w4**0.25, T_b4**0.25, T_g4**0.25]

# Function to calculate temperature without life
def calc_temp_no_life(L):
    alpha_p = alpha_g
    T4 = s_0 * L * (1 - alpha_p) / sigma
    return T4**0.25 - 273.15

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
t_span = (0, 1000) # Time span for simulation

# Luminosity range
L_values = np.linspace(0.0, 2.0, 1000)

# Run simulations for increasing luminosity
initial = [a_w0, a_b0]
steady_states_inc = []
temps_inc = []

for L in L_values:
    res = solve_ivp(DE, t_span, initial, args=(L,), dense_output=True, rtol=1e-8, atol=1e-8)
    
    final_aw = res.y[0, -1]
    final_ab = res.y[1, -1]
    final_ag = p - final_aw - final_ab
    steady_states_inc.append([L, final_aw, final_ab, final_ag, final_aw + final_ab])

    temps = calc_temps([final_aw, final_ab], L)
    planet_temp = temps[0] - 273.15
    white_temp = temps[1] - 273.15
    black_temp = temps[2] - 273.15
    no_life_temp = calc_temp_no_life(L)
    temps_inc.append([L, planet_temp, white_temp, black_temp, no_life_temp])

    if final_aw <= 0:
        final_aw = 0.01
    if final_ab <= 0:
        final_ab = 0.01

    initial = [final_aw, final_ab]

# Run simulations for decreasing luminosity
initial = [a_w0, a_b0]
steady_states_dec = []
temps_dec = []

for L in reversed(L_values):
    res = solve_ivp(DE, t_span, initial, args=(L,), dense_output=True, rtol=1e-8, atol=1e-8)
    
    final_aw = res.y[0, -1]
    final_ab = res.y[1, -1]
    final_ag = p - final_aw - final_ab
    steady_states_dec.append([L, final_aw, final_ab, final_ag, final_aw + final_ab])

    temps = calc_temps([final_aw, final_ab], L)
    planet_temp = temps[0] - 273.15
    white_temp = temps[1] - 273.15
    black_temp = temps[2] - 273.15
    no_life_temp = calc_temp_no_life(L)
    temps_dec.append([L, planet_temp, white_temp, black_temp, no_life_temp])


    if final_aw <= 0:
        final_aw = 0.01
    if final_ab <= 0:
        final_ab = 0.01

    initial = [final_aw, final_ab]

# Convert to numpy arrays
steady_states_inc = np.array(steady_states_inc)
temps_inc = np.array(temps_inc)
steady_states_dec = np.array(steady_states_dec)
temps_dec = np.array(temps_dec)

# Reverse to match L_values order
steady_states_dec = steady_states_dec[::-1]
temps_dec = temps_dec[::-1] 

# Create interactive plots
fig = plt.figure(figsize=(16, 10))
plt.suptitle('Fractions of daisies and temperatures as function of the luminosity', fontsize=16)

# Set height ratios
grid = GridSpec(2, 2, height_ratios=[3, 1])
ax1 = plt.subplot(grid[0, 0])
ax2 = plt.subplot(grid[0, 1])

# Left subplot for area fractions
ax1.grid(True)
ax1.set_xlim(0, 1.8)
ax1.set_ylim(0, 0.8)
ax1.set_xlabel('Luminosity')
ax1.set_ylabel('Area fractions')

# Plot all lines initially
line_white_inc, = ax1.plot(steady_states_inc[:, 0], steady_states_inc[:, 1], 'r-', label='White daisies (increasing L)')
line_white_dec, = ax1.plot(steady_states_dec[:, 0], steady_states_dec[:, 1], 'm-', label='White daisies (decreasing L)')
line_black_inc, = ax1.plot(steady_states_inc[:, 0], steady_states_inc[:, 2], 'b-', label='Black daisies (increasing L)')
line_black_dec, = ax1.plot(steady_states_dec[:, 0], steady_states_dec[:, 2], 'c-', label='Black daisies (decreasing L)')
line_total_inc, = ax1.plot(steady_states_inc[:, 0], steady_states_inc[:, 4], 'k-', label='Total daisies (increasing L)')
line_total_dec, = ax1.plot(steady_states_dec[:, 0], steady_states_dec[:, 4], 'g-', label='Total daisies (decreasing L)')

# Right subplot for temperatures
ax2.grid(True)
ax2.set_xlim(0, 2.0)
ax2.set_ylim(0, 100)
ax2.set_xlabel('Luminosity')
ax2.set_ylabel('Temperature (°C)')

# Temperature lines
line_no_life, = ax2.plot(L_values, [calc_temp_no_life(L) for L in L_values], 'k-', label='Temperature without life')
line_temp_inc, = ax2.plot(temps_inc[:, 0], temps_inc[:, 1], 'b-', label='Global temperature (increasing L)')
line_temp_dec, = ax2.plot(temps_dec[:, 0], temps_dec[:, 1], 'g-', label='Global temperature (decreasing L)')
line_opt_temp, = ax2.plot([0.0, 2.0], [T_opt-273.15, T_opt-273.15], 'y--', label='Optimal temperature')
line_white_temp_inc, = ax2.plot(temps_inc[:, 0], temps_inc[:, 2], 'c-', label='White daisies temperature')
line_black_temp_inc, = ax2.plot(temps_inc[:, 0], temps_inc[:, 3], 'm-', label='Black daisies temperature')

# Create legend boxes with checkboxes
ax_check1 = plt.subplot(grid[1, 0])
ax_check2 = plt.subplot(grid[1, 1])

# Remove axis ticks and labels for checkbox areas
ax_check1.axis('off')
ax_check2.axis('off')

# Define the labels for area fractions
labels1 = [
    'Area fraction of white daisies for increasing L',
    'Area fraction of white daisies for decreasing L',
    'Area fraction of black daisies for increasing L',
    'Area fraction of black daisies for decreasing L',
    'Total amount of daisies for increasing L',
    'Total amount of daisies for decreasing L'
]
lines1 = [line_white_inc, line_white_dec, line_black_inc, line_black_dec, line_total_inc, line_total_dec]

# Define the labels for temperatures
labels2 = [
    'Temperature without life',
    'Global temperature for increasing luminosity',
    'Global temperature for decreasing luminosity',
    'Optimal temperature',
    'White daisies local temperature',
    'Black daisies local temperature'
]
lines2 = [line_no_life, line_temp_inc, line_temp_dec, line_opt_temp, line_white_temp_inc, line_black_temp_inc]

# Create CheckButtons
check1 = CheckButtons(ax=ax_check1, labels=labels1, actives=[True for _ in labels1])
check2 = CheckButtons(ax=ax_check2, labels=labels2, actives=[True for _ in labels2])

# Toggle visibility
def func1(label):
    index = labels1.index(label)
    lines1[index].set_visible(not lines1[index].get_visible())
    plt.draw()

def func2(label):
    index = labels2.index(label)
    lines2[index].set_visible(not lines2[index].get_visible())
    plt.draw()

check1.on_clicked(func1)
check2.on_clicked(func2)

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()
#plt.savefig('Graphs.html', format='html')