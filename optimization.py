# optimization.py
import time
import numpy as np
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor
import itertools
from setup_data import OpticalSetup

def calculate_cavity_modes(L, C1, R2, wl):
    g1 = 1.0 - L * C1
    g2 = 1.0 - L / R2
    if not (0.001 <= g1 * g2 <= 0.999): 
        raise ValueError("Unstable cavity.")
        
    w1_sq = (L * wl / np.pi) * np.sqrt(g2 / (g1 * (1 - g1 * g2)))
    w2_sq = (L * wl / np.pi) * np.sqrt(g1 / (g2 * (1 - g1 * g2)))
    return np.sqrt(w1_sq), np.sqrt(w2_sq)

def get_beam_radius(q, wavelength):
    inv_q = 1.0 / q
    w = np.sqrt(wavelength / (-np.pi * np.imag(inv_q)))
    R = 1.0 / np.real(inv_q) if np.real(inv_q) != 0 else np.inf
    return w, R

def calculate_coupling_efficiency(w1, R1, w2, R2, wl):
    inv_R1 = 0.0 if np.isinf(R1) or R1 == 0 else 1.0 / R1
    inv_R2 = 0.0 if np.isinf(R2) or R2 == 0 else 1.0 / R2
    num = 4 * (w1**2) * (w2**2)
    den = (w1**2 + w2**2)**2 + ((np.pi * w1**2 * w2**2) / wl)**2 * (inv_R1 - inv_R2)**2
    return num / den

def get_window_shift(t, theta_deg, n):
    theta = np.deg2rad(theta_deg)
    return t * (1.0 - np.cos(theta) / np.sqrt(n**2 - np.sin(theta)**2))

def monte_carlo_tolerances(setup: OpticalSetup, focal_lengths: list, distances: list, samples: int = 500):
    np.random.seed(42) 
    
    wl_s = np.random.uniform(setup.wavelength - setup.tol_wl*1e-6, setup.wavelength + setup.tol_wl*1e-6, samples)
    L_s = np.random.uniform(setup.cavity_length - setup.tol_cl, setup.cavity_length + setup.tol_cl, samples)
    C1_s = np.random.uniform(-1.0/setup.tol_r1, 1.0/setup.tol_r1, samples)
    R2_s = np.random.uniform(setup.cavity_R2 - setup.tol_r2, setup.cavity_R2 + setup.tol_r2, samples)
    
    if setup.use_fiber_coupler:
        fcore_s = np.random.uniform(setup.fiber_core_mfd - setup.tol_fcore, setup.fiber_core_mfd + setup.tol_fcore, samples)
        fcoup_s = np.random.uniform(setup.coupler_f - setup.tol_coupler_f, setup.coupler_f + setup.tol_coupler_f, samples)
        w0_s = (fcoup_s * wl_s) / (np.pi * (fcore_s / 2.0))
    else:
        w0_s = np.random.uniform(setup.source_w0 - setup.tol_w0, setup.source_w0 + setup.tol_w0, samples)
        
    f_s = [np.random.uniform(f * (1 - setup.tol_f/100.0), f * (1 + setup.tol_f/100.0), samples) for f in focal_lengths]
    d_s = [np.random.uniform(d - setup.tol_pos, d + setup.tol_pos, samples) for d in distances]
    
    if setup.use_viewport:
        t_s = np.random.uniform(setup.window_thickness - setup.tol_thickness, setup.window_thickness + setup.tol_thickness, samples)
        ang_s = np.random.uniform(setup.window_angle - setup.tol_angle, setup.window_angle + setup.tol_angle, samples)
        shift_s = get_window_shift(t_s, ang_s, setup.window_n)
    else:
        shift_s = np.zeros(samples)
        
    worst_dev_f, worst_dev_c = 0.0, 0.0
    worst_eff = 1.0
    t_f_nom, t_c_nom = calculate_cavity_modes(setup.cavity_length, 0.0, setup.cavity_R2, setup.wavelength)
    
    for i in range(samples):
        try:
            t_f_i, t_c_i = calculate_cavity_modes(L_s[i], C1_s[i], R2_s[i], wl_s[i]) 
        except ValueError: continue
            
        q = 1j * (np.pi * (w0_s[i] ** 2)) / wl_s[i]
        for j in range(len(focal_lengths)):
            q = q + d_s[j][i]
            q = (1.0 * q) / ((-1.0 / f_s[j][i]) * q + 1.0)
            
        q = q + (d_s[-1][i] - shift_s[i])
        
        achieved_w_f, achieved_R_f = get_beam_radius(q, wl_s[i])
        achieved_w_c, _ = get_beam_radius(q + L_s[i], wl_s[i])
        
        dev_f = abs(achieved_w_f - t_f_nom)
        dev_c = abs(achieved_w_c - t_c_nom)
        
        target_R1 = 1.0/C1_s[i] if C1_s[i] != 0 else np.inf
        eff = calculate_coupling_efficiency(achieved_w_f, achieved_R_f, t_f_i, target_R1, wl_s[i])
        
        if dev_f > worst_dev_f: worst_dev_f = dev_f
        if dev_c > worst_dev_c: worst_dev_c = dev_c
        if eff < worst_eff: worst_eff = eff
        
    return worst_dev_f, worst_dev_c, worst_eff

def optimize_single_combination(args):
    variable_lens_combo, setup, target_w_flat, target_w_curved, start_time, max_time = args
    if time.time() - start_time > max_time: return None

    full_f = []
    var_idx = 0
    for f in setup.lenses:
        if f is not None: full_f.append(f)
        else:
            full_f.append(variable_lens_combo[var_idx])
            var_idx += 1

    shift = get_window_shift(setup.window_thickness, setup.window_angle, setup.window_n) if setup.use_viewport else 0.0

    def objective(distances):
        w0 = ((setup.coupler_f * setup.wavelength) / (np.pi * (setup.fiber_core_mfd / 2.0))) if setup.use_fiber_coupler else setup.source_w0
        q = 1j * (np.pi * (w0 ** 2)) / setup.wavelength
        
        for i in range(len(full_f)):
            q = q + distances[i]
            q = (1.0 * q) / ((-1.0 / full_f[i]) * q + 1.0)
            
        q = q + (distances[-1] - shift)
        w_flat, R_flat = get_beam_radius(q, setup.wavelength)
        eff = calculate_coupling_efficiency(w_flat, R_flat, target_w_flat, np.inf, setup.wavelength)
        return 1.0 - eff

    bounds = setup.distance_bounds
    initial_guess = [sum(b)/2 for b in bounds]
    res = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds, tol=1e-8)
    
    if res.success:
        w0 = ((setup.coupler_f * setup.wavelength) / (np.pi * (setup.fiber_core_mfd / 2.0))) if setup.use_fiber_coupler else setup.source_w0
        q = 1j * (np.pi * (w0 ** 2)) / setup.wavelength
        for i in range(len(full_f)):
            q = q + res.x[i]
            q = (1.0 * q) / ((-1.0 / full_f[i]) * q + 1.0)
            
        q = q + (res.x[-1] - shift)
        
        f_w_f, f_R_f = get_beam_radius(q, setup.wavelength)
        f_w_c, _ = get_beam_radius(q + setup.cavity_length, setup.wavelength)
        nom_eff = calculate_coupling_efficiency(f_w_f, f_R_f, target_w_flat, np.inf, setup.wavelength)
        
        worst_f, worst_c, worst_eff = monte_carlo_tolerances(setup, full_f, res.x)
        
        return {
            'focal_lengths': full_f, 'variable_lenses': variable_lens_combo, 'distances': res.x,
            'achieved_w_flat': f_w_f, 'achieved_w_curved': f_w_c, 'achieved_R_flat': f_R_f,
            'target_w_flat': target_w_flat, 'target_w_curved': target_w_curved,
            'nom_eff': nom_eff, 'worst_eff': worst_eff,
            'worst_dev_flat': worst_f, 'worst_dev_curved': worst_c, 
            'accuracy': 1.0 - nom_eff,
            'initial_w0': w0
        }
    return None

def run_multithreaded_optimization(setup: OpticalSetup, max_time: float, progress_callback=None):
    start_time = time.time()
    t_w_flat, t_w_curved = calculate_cavity_modes(setup.cavity_length, 0.0, setup.cavity_R2, setup.wavelength)
    
    num_variable_lenses = sum(1 for f in setup.lenses if f is None)
    variable_combinations = list(itertools.product(setup.available_lenses, repeat=num_variable_lenses))
    
    all_results = []
    total_tasks = len(variable_combinations)
    completed_tasks = 0
    tasks = [(combo, setup, t_w_flat, t_w_curved, start_time, max_time) for combo in variable_combinations]
    
    with ThreadPoolExecutor() as executor:
        results = executor.map(optimize_single_combination, tasks)
        for res in results:
            completed_tasks += 1
            if progress_callback: progress_callback(int((completed_tasks / total_tasks) * 100))
            if res is not None: all_results.append(res)
            if time.time() - start_time >= max_time: break
                
    all_results.sort(key=lambda x: x['accuracy']) 
    return all_results

def generate_beam_profile_data(setup: OpticalSetup, focal_lengths: list, distances: list):
    """Calculates continuous beam envelope strictly moving forward from source."""
    max_z = sum(distances) + setup.cavity_length
    z_coords = np.linspace(0, max_z, 400)
    
    if setup.use_fiber_coupler:
        w0 = (setup.coupler_f * setup.wavelength) / (np.pi * (setup.fiber_core_mfd / 2.0))
    else:
        w0 = setup.source_w0
        
    w_coords = np.zeros_like(z_coords)
    
    shift = get_window_shift(setup.window_thickness, setup.window_angle, setup.window_n) if setup.use_viewport else 0.0
    flat_z = sum(distances)
    vp_end = flat_z - setup.window_dist if setup.use_viewport else None
    vp_start = vp_end - setup.window_thickness if setup.use_viewport else None

    q_start = 1j * (np.pi * (w0 ** 2)) / setup.wavelength
    
    # Strictly forward propagation
    for i, z in enumerate(z_coords):
        q = q_start
        z_current = 0.0
        
        for j, d in enumerate(distances):
            if z <= z_current + d:
                dz = z - z_current
                
                # Apply gradual viewport optical path reduction if passing through it
                if setup.use_viewport and z > vp_start:
                    if z < vp_end:
                        partial_t = z - vp_start
                        dz -= get_window_shift(partial_t, setup.window_angle, setup.window_n)
                    else:
                        dz -= shift
                        
                q = q + dz
                w_coords[i], _ = get_beam_radius(q, setup.wavelength)
                break
            else:
                # Fully propagate through this segment
                seg_d = d
                if setup.use_viewport and j == len(distances) - 1:
                    seg_d -= shift
                    
                q = q + seg_d
                z_current += d
                
                if j < len(focal_lengths):
                    q = (1.0 * q) / ((-1.0 / focal_lengths[j]) * q + 1.0)
                    
        # In cavity
        if z > flat_z:
            dz = z - flat_z
            q = q + dz
            w_coords[i], _ = get_beam_radius(q, setup.wavelength)
            
    lens_z = np.cumsum(distances[:-1])
    curved_z = flat_z + setup.cavity_length
    
    return z_coords, w_coords, lens_z, flat_z, curved_z, vp_start, vp_end