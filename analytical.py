import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from sympy import symbols, solve, Eq, solveset, S, lambdify

# Define symbolic variables
L, aw, ab, ag, T, Tw, Tb, Tg, alpha_p = symbols('L aw ab ag T Tw Tb Tg alpha_p', real=True, positive=True)

# Define constants
alpha_w = 0.75  # albedo of white daisies (high reflectivity)
alpha_b = 0.25  # albedo of black daisies (high absorption)
alpha_g = 0.5   # albedo of bare ground (by convention)
gamma = 0.3     # death rate
Topt = 295.5    # optimal temperature in K
k = 1/(17.5**2) # parabolic width parameter
S0 = 917        # average solar energy flux in W/m^2
sigma = 5.67e-8 # Stefan-Boltzmann constant in W/m^2/K^4
q = 2.06e9      # heat transfer coefficient in K^4

print("Step 1: Define the system parameters and equations")
print(f"Constants: alpha_w = {alpha_w}, alpha_b = {alpha_b}, alpha_g = {alpha_g}")
print(f"gamma = {gamma}, Topt = {Topt} K, k = {k} K^-2")
print(f"S0 = {S0} W/m^2, sigma = {sigma} W/m^2/K^4, q = {q} K^4")

# Define the birth rate function β(T)
def beta(T_val):
    if abs(T_val - Topt) < 1/np.sqrt(k):
        return 1 - k * (T_val - Topt)**2
    else:
        return 0

print("\nStep 2: Define analytical expressions for the system")

# Express ag in terms of aw and ab
ag_expr = 1 - aw - ab  # assuming p=1 (fully fertile ground)
print(f"Area fraction of bare ground: ag = {ag_expr}")

# Planetary albedo
alpha_p_expr = aw * alpha_w + ab * alpha_b + (1 - aw - ab) * alpha_g
print(f"Planetary albedo: alpha_p = {alpha_p_expr}")

# Simplify the albedo expression
alpha_p_simplified = sp.expand(alpha_p_expr)
print(f"Simplified planetary albedo: alpha_p = {alpha_p_simplified}")

# Express T in terms of alpha_p and L using the planetary heat balance
T_expr = sp.sympify(f"(S0*L*(1-alpha_p)/sigma)**(1/4)")
print(f"Planetary temperature: T = {T_expr}")

# Express local temperatures in terms of T and alpha_p
Tw_expr = sp.sympify(f"(q*(alpha_p-{alpha_w})+T**4)**(1/4)")
Tb_expr = sp.sympify(f"(q*(alpha_p-{alpha_b})+T**4)**(1/4)")
Tg_expr = sp.sympify(f"(q*(alpha_p-{alpha_g})+T**4)**(1/4)")

print(f"White daisy local temperature: Tw = {Tw_expr}")
print(f"Black daisy local temperature: Tb = {Tb_expr}")
print(f"Bare ground temperature: Tg = {Tg_expr}")

print("\nStep 3: Solve for steady state solutions")
print("At steady state, daw/dt = dab/dt = 0, which gives us:")

# For steady state, we need daw/dt = dab/dt = 0
# From equation (1), this means:
# aw[ag*β(Tw)-γ] = 0, which gives either aw = 0 or ag*β(Tw) = γ
# ab[ag*β(Tb)-γ] = 0, which gives either ab = 0 or ag*β(Tb) = γ

print("For white daisies (aw):")
print("Either aw = 0 or ag*Beta(Tw) = gamma")
print("For black daisies (ab):")
print("Either ab = 0 or ag*Beta(Tb) = gamma")

print("\nStep 4: Create a numerical solver to find steady state solutions")
print("Since the full system is complex, we'll solve it numerically for different luminosity values")

def compute_steady_state(L_val, initial_guess=(0.3, 0.3)):
    """
    Compute the steady state solution for a given luminosity L
    
    Parameters:
    L_val: Luminosity value
    initial_guess: Initial guess for (aw, ab)
    
    Returns:
    (aw, ab, T, Tw, Tb) at steady state
    """
    # Define the residual function for the steady state
    def residual(x):
        aw_val, ab_val = x
        
        # Check bounds
        if aw_val < 0 or ab_val < 0 or aw_val + ab_val > 1:
            return [1e10, 1e10]  # Return large residual for invalid solutions
        
        ag_val = 1 - aw_val - ab_val
        alpha_p_val = aw_val * alpha_w + ab_val * alpha_b + ag_val * alpha_g
        
        # Compute temperatures
        T_val = (S0 * L_val * (1 - alpha_p_val) / sigma) ** 0.25
        Tw_val = (q * (alpha_p_val - alpha_w) + T_val**4) ** 0.25
        Tb_val = (q * (alpha_p_val - alpha_b) + T_val**4) ** 0.25
        
        # Compute birth rates
        beta_w = beta(Tw_val)
        beta_b = beta(Tb_val)
        
        # Compute residuals from steady state conditions
        r1 = aw_val * (ag_val * beta_w - gamma)
        r2 = ab_val * (ag_val * beta_b - gamma)
        
        return [r1, r2]
    
    # Solve the system
    from scipy.optimize import fsolve
    solution, info = fsolve(residual, initial_guess, full_output=True)
    
    if info['fvec'].max() > 1e-6:
        # Try different initial conditions if the solution didn't converge
        attempts = [
            (0.0, 0.0), (0.6, 0.0), (0.0, 0.6), 
            (0.3, 0.3), (0.4, 0.2), (0.2, 0.4)
        ]
        
        for guess in attempts:
            solution, info = fsolve(residual, guess, full_output=True)
            if info['fvec'].max() < 1e-6:
                break
    
    aw_val, ab_val = solution
    aw_val = max(0, min(aw_val, 1))  # Ensure aw is between 0 and 1
    ab_val = max(0, min(ab_val, 1))  # Ensure ab is between 0 and 1
    
    if aw_val + ab_val > 1:
        # Normalize if sum exceeds 1
        total = aw_val + ab_val
        aw_val /= total
        ab_val /= total
    
    # Calculate derived quantities
    ag_val = 1 - aw_val - ab_val
    alpha_p_val = aw_val * alpha_w + ab_val * alpha_b + ag_val * alpha_g
    T_val = (S0 * L_val * (1 - alpha_p_val) / sigma) ** 0.25
    Tw_val = (q * (alpha_p_val - alpha_w) + T_val**4) ** 0.25
    Tb_val = (q * (alpha_p_val - alpha_b) + T_val**4) ** 0.25
    
    return aw_val, ab_val, T_val, Tw_val, Tb_val

# Compute solutions across a range of luminosity values
L_values = np.linspace(0.5, 1.6, 100)
results = []

for L_val in L_values:
    # Use previous result as initial guess if available
    if results:
        initial_guess = (results[-1][0], results[-1][1])
    else:
        initial_guess = (0.3, 0.3)
    
    try:
        result = compute_steady_state(L_val, initial_guess)
        results.append(result)
    except:
        # If the solver fails, append a placeholder
        results.append((0, 0, 0, 0, 0))

# Convert results to arrays for plotting
results_array = np.array(results)
aw_values = results_array[:, 0]
ab_values = results_array[:, 1]
T_values = results_array[:, 2]
Tw_values = results_array[:, 3]
Tb_values = results_array[:, 4]

print("\nStep 5: Analyze the results and create plots")

# Plot area fractions
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(L_values, aw_values, 'w-', label='White daisies', markerfacecolor='none', markeredgecolor='black')
plt.plot(L_values, ab_values, 'k-', label='Black daisies')
plt.plot(L_values, 1 - aw_values - ab_values, 'gray', label='Bare ground')
plt.axvline(x=1.0, color='r', linestyle='--', alpha=0.5)
plt.grid(True, alpha=0.3)
plt.xlabel('Luminosity (L)')
plt.ylabel('Area fraction')
plt.legend()
plt.title('Daisyworld: Area Fractions vs. Luminosity')

# Plot temperatures
plt.subplot(2, 1, 2)
plt.plot(L_values, T_values, 'r-', label='Planet T')
plt.plot(L_values, Tw_values, 'b-', label='White daisy T')
plt.plot(L_values, Tb_values, 'k-', label='Black daisy T')
plt.axhline(y=Topt, color='g', linestyle='--', label=f'Optimal T ({Topt} K)')
plt.axvline(x=1.0, color='r', linestyle='--', alpha=0.5)
plt.grid(True, alpha=0.3)
plt.xlabel('Luminosity (L)')
plt.ylabel('Temperature (K)')
plt.legend()
plt.title('Daisyworld: Temperatures vs. Luminosity')

plt.tight_layout()

print("\nStep 6: Find analytical steady-state solutions for specific cases")

# Case 1: Only white daisies (ab = 0)
print("\nCase 1: Only white daisies (ab = 0)")
# Substitute ab = 0 in our expressions
ab_val = 0
alpha_p_white = alpha_p_expr.subs(ab, ab_val)
print(f"Planetary albedo: alpha_p = {alpha_p_white}")

# For steady state with white daisies:
# ag*β(Tw) = γ where ag = 1 - aw - ab = 1 - aw
# (1 - aw)*β(Tw) = γ
# β(Tw) = γ/(1-aw)

print("At steady state: (1 - aw)*Beta(Tw) = gamma")
print(f"This gives: Beta(Tw) = {gamma}/(1-aw)")

# Case 2: Only black daisies (aw = 0)
print("\nCase 2: Only black daisies (aw = 0)")
# Substitute aw = 0 in our expressions
aw_val = 0
alpha_p_black = alpha_p_expr.subs(aw, aw_val)
print(f"Planetary albedo: alpha_p = {alpha_p_black}")

# For steady state with black daisies:
# ag*β(Tb) = γ where ag = 1 - aw - ab = 1 - ab
# (1 - ab)*β(Tb) = γ
# β(Tb) = γ/(1-ab)

print("At steady state: (1 - ab)*Beta(Tb) = gamma")
print(f"This gives: Beta(Tb) = {gamma}/(1-ab)")

# Case 3: Mixed state (both daisies present)
print("\nCase 3: Mixed state (both white and black daisies present)")
print("For a mixed state, we need:")
print("ag*Beta(Tw) = gamma and ag*Beta(Tb) = gamma")
print("This means Beta(Tw) = Beta(Tb), which occurs when Tw = Tb")

# The mixed state requires both species to have equal local temperatures:
print("This special condition occurs when the local temperatures are equal: Tw = Tb")
print("This balance is only achieved at specific luminosity values")

print("\nStep 7: Summarize key insights from the Daisyworld model")
print("1. Self-regulation: The system regulates planetary temperature through albedo changes")
print("2. Biotic regulation: At L=1, planetary temperature is kept close to Topt")
print("3. Hysteresis: Different system states can exist for the same luminosity value")
print("4. Complementary roles: Black daisies dominate at low luminosity, white at high luminosity")
print("5. Habitable range: Daisies extend the habitable luminosity range beyond abiotic conditions")

plt.savefig('daisyworld_results.png')
plt.show()