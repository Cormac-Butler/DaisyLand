from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Slider, Button
from matplotlib.gridspec import GridSpec

# Global variables to store simulation data
steady_states_inc = None
steady_states_dec = None
temps_inc = None
temps_dec = None

# Function defining the system of differential equations
def DE(t, y, L, alpha_w, alpha_b, gamma, s_0, q, T_opt, k, alpha_g):
    a_w, a_b = y
    a_g = p - a_w - a_b
    temps = calc_temps(y, L, alpha_w, alpha_b, s_0, q, alpha_g)
    beta_w = beta_T(temps[1], T_opt, k)
    beta_b = beta_T(temps[2], T_opt, k)
    DE1 = a_w * (a_g * beta_w - gamma)
    DE2 = a_b * (a_g * beta_b - gamma)
    return [DE1, DE2]

# Function to calculate growth rate based on temperature
def beta_T(T, T_opt, k):
    return 1 - k * (T - T_opt) ** 2 if abs(T - T_opt) < (1 / np.sqrt(k)) else 0

# Function to calculate temperatures
def calc_temps(y, L, alpha_w, alpha_b, s_0, q, alpha_g):
    a_w, a_b = y
    a_g = p - a_w - a_b
    alpha_p = a_w * alpha_w + a_b * alpha_b + a_g * alpha_g
    T4 = s_0 * L * (1 - alpha_p) / sigma
    T_w4 = q * (alpha_p - alpha_w) + T4
    T_b4 = q * (alpha_p - alpha_b) + T4
    T_g4 = q * (alpha_p - alpha_g) + T4
    return [T4**0.25, T_w4**0.25, T_b4**0.25, T_g4**0.25]

# Function to calculate temperature without life
def calc_temp_no_life(L, s_0, alpha_g):
    alpha_p = alpha_g
    T4 = s_0 * L * (1 - alpha_p) / sigma
    return T4**0.25 - 273.15

# Simulation function
def simulate(alpha_w, alpha_b, gamma, s_0, q, T_1, T_2, alpha_g):
    global steady_states_inc, steady_states_dec, temps_inc, temps_dec
    T_opt = (T_1 + T_2) / 2
    k = 1 / ((T_2 - T_1) / 2) ** 2
    initial = [a_w0, a_b0]
    steady_states_inc = []
    temps_inc = []
    for L in L_values:
        res = solve_ivp(DE, t_span, initial, args=(L, alpha_w, alpha_b, gamma, s_0, q, T_opt, k, alpha_g),
                        dense_output=True, rtol=1e-8, atol=1e-8)
        final_aw = res.y[0, -1]
        final_ab = res.y[1, -1]
        final_ag = p - final_aw - final_ab
        steady_states_inc.append([L, final_aw, final_ab, final_ag, final_aw + final_ab])
        temps = calc_temps([final_aw, final_ab], L, alpha_w, alpha_b, s_0, q, alpha_g)
        temps_inc.append([L, temps[0] - 273.15, temps[1] - 273.15, temps[2] - 273.15, calc_temp_no_life(L, s_0, alpha_g)])
        initial = [max(final_aw, 0.01), max(final_ab, 0.01)]

    initial = [a_w0, a_b0]
    steady_states_dec = []
    temps_dec = []
    for L in reversed(L_values):
        res = solve_ivp(DE, t_span, initial, args=(L, alpha_w, alpha_b, gamma, s_0, q, T_opt, k, alpha_g),
                        dense_output=True, rtol=1e-8, atol=1e-8)
        final_aw = res.y[0, -1]
        final_ab = res.y[1, -1]
        final_ag = p - final_aw - final_ab
        steady_states_dec.append([L, final_aw, final_ab, final_ag, final_aw + final_ab])
        temps = calc_temps([final_aw, final_ab], L, alpha_w, alpha_b, s_0, q, alpha_g)
        temps_dec.append([L, temps[0] - 273.15, temps[1] - 273.15, temps[2] - 273.15, calc_temp_no_life(L, s_0, alpha_g)])
        initial = [max(final_aw, 0.01), max(final_ab, 0.01)]

    steady_states_inc = np.array(steady_states_inc)
    temps_inc = np.array(temps_inc)
    steady_states_dec = np.array(steady_states_dec[::-1])
    temps_dec = np.array(temps_dec[::-1])
    return steady_states_inc, steady_states_dec, temps_inc, temps_dec

# Declaring Variables
sigma = 5.67e-8
s_0 = 917
q = 2.06e9
T_1 = 278
T_2 = 313
alpha_g = 0.5
alpha_w = 0.75
alpha_b = 0.25
p = 1
gamma = 0.3
a_w0 = 0.01
a_b0 = 0.01
t_span = (0, 1000)
L_values = np.linspace(0.0, 2.0, 100)

# Initial simulation
simulate(alpha_w, alpha_b, gamma, s_0, q, T_1, T_2, alpha_g)

# Create interactive Matplotlib figure
fig = plt.figure(figsize=(16, 14))
plt.suptitle('Fractions of daisies and temperatures as function of the luminosity', fontsize=16)

# Adjust layout for sliders and buttons
fig.subplots_adjust(bottom=0.4)
grid = GridSpec(2, 2, height_ratios=[3, 1])
ax1 = plt.subplot(grid[0, 0])
ax2 = plt.subplot(grid[0, 1])
ax_check1 = plt.subplot(grid[1, 0])
ax_check2 = plt.subplot(grid[1, 1])

# Area fractions plot
ax1.grid(True)
ax1.set_xlim(0, 1.8)
ax1.set_ylim(0, 0.8)
ax1.set_xlabel('Luminosity')
ax1.set_ylabel('Area fractions')
line_white_inc, = ax1.plot(steady_states_inc[:, 0], steady_states_inc[:, 1], 'r-', label='White daisies (increasing L)')
line_white_dec, = ax1.plot(steady_states_dec[:, 0], steady_states_dec[:, 1], 'm-', label='White daisies (decreasing L)')
line_black_inc, = ax1.plot(steady_states_inc[:, 0], steady_states_inc[:, 2], 'b-', label='Black daisies (increasing L)')
line_black_dec, = ax1.plot(steady_states_dec[:, 0], steady_states_dec[:, 2], 'c-', label='Black daisies (decreasing L)')
line_total_inc, = ax1.plot(steady_states_inc[:, 0], steady_states_inc[:, 4], 'k-', label='Total daisies (increasing L)')
line_total_dec, = ax1.plot(steady_states_dec[:, 0], steady_states_dec[:, 4], 'g-', label='Total daisies (decreasing L)')

# Temperatures plot
ax2.grid(True)
ax2.set_xlim(0, 2.0)
ax2.set_ylim(0, 100)
ax2.set_xlabel('Luminosity')
ax2.set_ylabel('Temperature (°C)')
line_no_life, = ax2.plot(L_values, [calc_temp_no_life(L, s_0, alpha_g) for L in L_values], 'k-', label='Temperature without life')
line_temp_inc, = ax2.plot(temps_inc[:, 0], temps_inc[:, 1], 'b-', label='Global temperature (increasing L)')
line_temp_dec, = ax2.plot(temps_dec[:, 0], temps_dec[:, 1], 'g-', label='Global temperature (decreasing L)')
line_opt_temp, = ax2.plot([0.0, 2.0], [((T_1 + T_2) / 2) - 273.15, ((T_1 + T_2) / 2) - 273.15], 'y--', label='Optimal temperature')
line_white_temp_inc, = ax2.plot(temps_inc[:, 0], temps_inc[:, 2], 'c-', label='White daisies temperature')
line_black_temp_inc, = ax2.plot(temps_inc[:, 0], temps_inc[:, 3], 'm-', label='Black daisies temperature')

# Checkboxes
ax_check1.axis('off')
ax_check2.axis('off')
labels1 = [
    'Area fraction of white daisies for increasing L',
    'Area fraction of white daisies for decreasing L',
    'Area fraction of black daisies for increasing L',
    'Area fraction of black daisies for decreasing L',
    'Total amount of daisies for increasing L',
    'Total amount of daisies for decreasing L'
]
lines1 = [line_white_inc, line_white_dec, line_black_inc, line_black_dec, line_total_inc, line_total_dec]
labels2 = [
    'Temperature without life',
    'Global temperature for increasing luminosity',
    'Global temperature for decreasing luminosity',
    'Optimal temperature',
    'White daisies local temperature',
    'Black daisies local temperature'
]
lines2 = [line_no_life, line_temp_inc, line_temp_dec, line_opt_temp, line_white_temp_inc, line_black_temp_inc]
check1 = CheckButtons(ax_check1, labels1, [True] * len(labels1))
check2 = CheckButtons(ax_check2, labels2, [True] * len(labels2))

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

# Add sliders
slider_positions = {
    's_0': [0.1, 0.35, 0.8, 0.03],
    'q': [0.1, 0.30, 0.8, 0.03],
    'T_1': [0.1, 0.25, 0.8, 0.03],
    'T_2': [0.1, 0.20, 0.8, 0.03],
    'alpha_g': [0.1, 0.15, 0.8, 0.03],
    'alpha_w': [0.1, 0.10, 0.8, 0.03],
    'alpha_b': [0.1, 0.05, 0.8, 0.03],
    'gamma': [0.1, 0.00, 0.8, 0.03]
}

sliders = {}
sliders['s_0'] = Slider(fig.add_axes(slider_positions['s_0']), 's_0', 500, 1500, valinit=s_0)
sliders['q'] = Slider(fig.add_axes(slider_positions['q']), 'q', 1e8, 1e10, valinit=q)
sliders['T_1'] = Slider(fig.add_axes(slider_positions['T_1']), 'T_1', 250, 300, valinit=T_1)
sliders['T_2'] = Slider(fig.add_axes(slider_positions['T_2']), 'T_2', 300, 350, valinit=T_2)
sliders['alpha_g'] = Slider(fig.add_axes(slider_positions['alpha_g']), 'alpha_g', 0.0, 1.0, valinit=alpha_g)
sliders['alpha_w'] = Slider(fig.add_axes(slider_positions['alpha_w']), 'alpha_w', 0.0, 1.0, valinit=alpha_w)
sliders['alpha_b'] = Slider(fig.add_axes(slider_positions['alpha_b']), 'alpha_b', 0.0, 1.0, valinit=alpha_b)
sliders['gamma'] = Slider(fig.add_axes(slider_positions['gamma']), 'gamma', 0.0, 1.0, valinit=gamma)

# Add update button
ax_update = fig.add_axes([0.4, 0.40, 0.2, 0.03])
update_button = Button(ax_update, 'Update')

# Update function
def update(event):
    global steady_states_inc, steady_states_dec, temps_inc, temps_dec
    s_0_new = sliders['s_0'].val
    q_new = sliders['q'].val
    T_1_new = sliders['T_1'].val
    T_2_new = sliders['T_2'].val
    alpha_g_new = sliders['alpha_g'].val
    alpha_w_new = sliders['alpha_w'].val
    alpha_b_new = sliders['alpha_b'].val
    gamma_new = sliders['gamma'].val
    simulate(alpha_w_new, alpha_b_new, gamma_new, s_0_new, q_new, T_1_new, T_2_new, alpha_g_new)
    line_white_inc.set_data(steady_states_inc[:, 0], steady_states_inc[:, 1])
    line_white_dec.set_data(steady_states_dec[:, 0], steady_states_dec[:, 1])
    line_black_inc.set_data(steady_states_inc[:, 0], steady_states_inc[:, 2])
    line_black_dec.set_data(steady_states_dec[:, 0], steady_states_dec[:, 2])
    line_total_inc.set_data(steady_states_inc[:, 0], steady_states_inc[:, 4])
    line_total_dec.set_data(steady_states_dec[:, 0], steady_states_dec[:, 4])
    line_no_life.set_data(L_values, [calc_temp_no_life(L, s_0_new, alpha_g_new) for L in L_values])
    line_temp_inc.set_data(temps_inc[:, 0], temps_inc[:, 1])
    line_temp_dec.set_data(temps_dec[:, 0], temps_dec[:, 1])
    T_opt_new = (T_1_new + T_2_new) / 2
    line_opt_temp.set_data([0.0, 2.0], [T_opt_new - 273.15, T_opt_new - 273.15])
    line_white_temp_inc.set_data(temps_inc[:, 0], temps_inc[:, 2])
    line_black_temp_inc.set_data(temps_inc[:, 0], temps_inc[:, 3])
    fig.canvas.draw_idle()

update_button.on_clicked(update)

# Add save buttons
ax_save_area = fig.add_axes([0.1, 0.40, 0.2, 0.03])
ax_save_temp = fig.add_axes([0.7, 0.40, 0.2, 0.03])
save_area_button = Button(ax_save_area, 'Save Area Fractions')
save_temp_button = Button(ax_save_temp, 'Save Temperatures')

# Save functions
def save_area_fractions(event):
    fig_save = plt.figure()
    ax_save = fig_save.add_subplot(111)
    if line_white_inc.get_visible():
        ax_save.plot(steady_states_inc[:, 0], steady_states_inc[:, 1], 'r-', label='White daisies (increasing L)')
    if line_white_dec.get_visible():
        ax_save.plot(steady_states_dec[:, 0], steady_states_dec[:, 1], 'm-', label='White daisies (decreasing L)')
    if line_black_inc.get_visible():
        ax_save.plot(steady_states_inc[:, 0], steady_states_inc[:, 2], 'b-', label='Black daisies (increasing L)')
    if line_black_dec.get_visible():
        ax_save.plot(steady_states_dec[:, 0], steady_states_dec[:, 2], 'c-', label='Black daisies (decreasing L)')
    if line_total_inc.get_visible():
        ax_save.plot(steady_states_inc[:, 0], steady_states_inc[:, 4], 'k-', label='Total daisies (increasing L)')
    if line_total_dec.get_visible():
        ax_save.plot(steady_states_dec[:, 0], steady_states_dec[:, 4], 'g-', label='Total daisies (decreasing L)')
    ax_save.set_xlabel('Luminosity')
    ax_save.set_ylabel('Area fractions')
    ax_save.legend()
    fig_save.savefig('area_fractions.png')
    plt.close(fig_save)

def save_temperatures(event):
    fig_save = plt.figure()
    ax_save = fig_save.add_subplot(111)
    if line_no_life.get_visible():
        ax_save.plot(L_values, [calc_temp_no_life(L, s_0, alpha_g) for L in L_values], 'k-', label='Temperature without life')
    if line_temp_inc.get_visible():
        ax_save.plot(temps_inc[:, 0], temps_inc[:, 1], 'b-', label='Global temperature (increasing L)')
    if line_temp_dec.get_visible():
        ax_save.plot(temps_dec[:, 0], temps_dec[:, 1], 'g-', label='Global temperature (decreasing L)')
    if line_opt_temp.get_visible():
        ax_save.plot([0.0, 2.0], [((T_1 + T_2) / 2) - 273.15, ((T_1 + T_2) / 2) - 273.15], 'y--', label='Optimal temperature')
    if line_white_temp_inc.get_visible():
        ax_save.plot(temps_inc[:, 0], temps_inc[:, 2], 'c-', label='White daisies temperature')
    if line_black_temp_inc.get_visible():
        ax_save.plot(temps_inc[:, 0], temps_inc[:, 3], 'm-', label='Black daisies temperature')
    ax_save.set_xlabel('Luminosity')
    ax_save.set_ylabel('Temperature (°C)')
    ax_save.legend()
    fig_save.savefig('temperatures.png')
    plt.close(fig_save)

save_area_button.on_clicked(save_area_fractions)
save_temp_button.on_clicked(save_temperatures)

plt.show()