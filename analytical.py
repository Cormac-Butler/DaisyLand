import sympy as sp
import numpy as np

# Define time variable
t = sp.Symbol('t')

# Define functions for area fractions
aw = sp.Function('a_w')(t)
ab = sp.Function('a_b')(t)

# Define parameters
ag, Tw, Tb, Topt, k, gamma = sp.symbols('a_g T_w T_b T_opt k gamma', real=True, positive=True)

# Define birth rate function β(T)
beta_T = lambda T: sp.Piecewise((1 - k * (T - Topt), sp.Abs(T - Topt) < 1/(2*k)), (0, True))

# Define the differential equations
daw_dt = sp.Eq(aw.diff(t), aw * (ag * beta_T(Tw) - gamma))
dab_dt = sp.Eq(ab.diff(t), ab * (ag * beta_T(Tb) - gamma))

# Solve the differential equations with initial conditions
a_w0, a_b0 = 0.01, 0.01
solution_aw = sp.dsolve(daw_dt, aw, ics={aw.subs(t, 0): a_w0})
solution_ab = sp.dsolve(dab_dt, ab, ics={ab.subs(t, 0): a_b0})

# Declare fixed variables
σ = 5.67e-8  # Boltzmann constant
S_0 = 917  # Solar irradiance
q = 2.06e9  # Heat transfer coefficient
T_1, T_2 = 278, 313  # Temperature bounds
Topt = (T_1 + T_2) / 2  # Optimal Temperature
k = 1 / ((T_2 - T_1) / 2) ** 2  # Parabolic width
alpha_g, alpha_w, alpha_b = 0.5, 0.75, 0.25  # Albedos
p = 1  # Fraction of fertile land
gamma = 0.3  # Death rate

# Define simulation parameters
t_span = (0, 1000)  # Time span for simulation
L_values = np.linspace(0.0, 2.0, 1000)  # Luminosity range
initial = [a_w0, a_b0]  # Initial conditions

# Print the equations and solutions
print("Differential equations:")
print("d(aw)/dt =", daw_dt)
print("d(ab)/dt =", dab_dt)
print("\nGeneral solutions:")
print("aw(t) =", solution_aw)
print("ab(t) =", solution_ab)
