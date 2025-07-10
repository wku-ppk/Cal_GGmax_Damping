import numpy as np
import os
import matplotlib.pyplot as plt
import sys
import math
from scipy.signal import argrelmin

def load_cyclic_stress_data(filename):
    if not os.path.isfile(filename):
        sys.exit(f"[ERROR] File not found: {filename}")

    def find_data_start(lines):
        for i, line in enumerate(lines):
            try:
                [float(x) for x in line.strip().split(',')]
                return i
            except ValueError:
                continue
        raise ValueError(f"No numeric data found in '{filename}'.")

    with open(filename, 'r') as f:
        lines = f.readlines()
    start_row = find_data_start(lines)

    try:
        data = np.loadtxt(filename, delimiter=',', skiprows=start_row)
    except Exception as e:
        sys.exit(f"[ERROR] Failed to load numeric data: {e}")

    column_names = [
        'step', 'time', 'z_force', 'z_position', 'z_disp', 'void_ratio',
        'shear_strain', 's_xx', 's_yy', 's_zz', 's_xy', 's_xz', 's_yz'
    ]

    if data.shape[1] != len(column_names):
        print(f"[WARNING] Column count mismatch: expected {len(column_names)}, got {data.shape[1]}")

    columns = {name: data[:, i] for i, name in enumerate(column_names)}

    print(f"[INFO] Loaded {data.shape[0]} rows from '{filename}' (data starts at line {start_row + 1})")
    return data, columns

def detect_cycles_by_minima(shear_strain):
    minima_indices = argrelmin(shear_strain)[0]
    if minima_indices.size == 0:
        return []
    return [(minima_indices[i], minima_indices[i + 1]) for i in range(len(minima_indices) - 1)]

def compute_secant_modulus(shear_strain, stress):
    i_min = np.argmin(shear_strain)
    i_max = np.argmax(shear_strain)
    gamma_min = shear_strain[i_min]
    gamma_max = shear_strain[i_max]
    tau_min = stress[i_min]
    tau_max = stress[i_max]
    delta_gamma = gamma_max - gamma_min
    delta_tau = tau_max - tau_min
    if delta_gamma == 0:
        return 0.0, (gamma_min, tau_min), (gamma_max, tau_max)
    return delta_tau / delta_gamma, (gamma_min, tau_min), (gamma_max, tau_max)

def compute_damping_ratio(shear_strain, stress):
    """
    Computes damping ratio based on Itasca FLAC FISH script logic.
    Assumes full cycle data is provided (one complete hysteresis loop).
    """

    e = shear_strain
    t = stress

    n = len(e)
    if n < 4:
        return 0.0  # not enough points

    # Step 1: Find max/min strain and stress
    emax = np.max(e)
    emin = np.min(e)
    tmax = np.max(t)
    tmin = np.min(t)

    delta_gamma = emax - emin
    delta_tau = tmax - tmin

    # Secant modulus normalized (you can scale by given Gmax if available)
    slope = delta_tau / delta_gamma if delta_gamma != 0 else 0

    # Step 2: Split the cycle at midpoint index
    mid_idx = np.argmin(np.abs(e))  # closest to zero strain
    Tbase = t[mid_idx]

    # Step 3: Integrate lower loop area (strain decreasing)
    Lsum = 0.0
    for i in range(0, mid_idx):
        meanT = 0.5 * (t[i] + t[i + 1])
        Lsum += (e[i] - e[i + 1]) * (meanT - Tbase)

    # Step 4: Integrate upper loop area (strain increasing)
    Usum = 0.0
    for i in range(mid_idx, n - 1):
        meanT = 0.5 * (t[i] + t[i + 1])
        Usum += (e[i + 1] - e[i]) * (meanT - Tbase)

    # Step 5: Energy difference
    Wdiff = Usum - Lsum
    Senergy = 0.5 * abs(e[0] * (t[0] - Tbase))  # or another reference point

    if Senergy == 0:
        return 0.0

    Drat = Wdiff / (4.0 * np.pi * Senergy)
    return Drat

def plot_full_and_individual_cycles(shear_strain, s_xz, cycle_bounds, label_prefix="Cycle"):
    num_cycles = len(cycle_bounds)
    total_plots = num_cycles + 1
    cols = 3
    rows = math.ceil(total_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    axes[0].plot(shear_strain, s_xz, color='black', label='Full Data')
    axes[0].set_title('Full Hysteresis Curve')
    axes[0].set_xlabel("Shear Strain")
    axes[0].set_ylabel("s_xz Stress")
    axes[0].grid(True)
    axes[0].legend()

    for i, (start, end) in enumerate(cycle_bounds):
        gamma = shear_strain[start:end+1]
        tau = s_xz[start:end+1]
        gsec, pt1, pt2 = compute_secant_modulus(gamma, tau)
        damping = compute_damping_ratio(gamma, tau)

        ax = axes[i + 1]
        ax.plot(gamma, tau, label=f"{label_prefix} {i+1}")
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'r--', label='Secant Line')
        ax.set_title(f"{label_prefix} {i+1}\nGsec={gsec:.2e}, D={damping:.3f}")
        ax.set_xlabel("Shear Strain")
        ax.set_ylabel("s_xz Stress")
        ax.grid(True)
        ax.legend()

    for j in range(total_plots, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Full and Individual Hysteresis Cycles", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 ak.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]
    data, cols = load_cyclic_stress_data(filename)
    shear_strain = cols['shear_strain']
    s_xz = cols['s_xz']
    cycle_bounds = detect_cycles_by_minima(shear_strain)
    plot_full_and_individual_cycles(shear_strain, s_xz, cycle_bounds)