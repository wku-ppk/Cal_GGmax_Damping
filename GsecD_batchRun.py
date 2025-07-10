import os
import re
import numpy as np
from scipy.signal import argrelmin
from scipy.integrate import trapezoid

def load_cyclic_stress_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        try:
            _ = [float(x) for x in line.strip().split(',')]
            start_row = i
            break
        except ValueError:
            continue
    else:
        raise ValueError(f"No numeric data found in {filename}")

    data = np.loadtxt(filename, delimiter=',', skiprows=start_row)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    ncols = data.shape[1]
    if ncols == 9:
        names = ['step', 'time', 'shear_strain', 'void_ratio', 'z_force', 'z_disp', 's_zz', 's_xz', 's_yz']
    elif ncols == 13:
        names = ['step', 'time', 'z_force', 'z_position', 'z_disp', 'void_ratio',
                 'shear_strain', 's_xx', 's_yy', 's_zz', 's_xy', 's_xz', 's_yz']
    else:
        raise ValueError(f"Unexpected number of columns ({ncols}) in {filename}")

    return {name: data[:, i] for i, name in enumerate(names)}

def detect_cycles_by_minima(shear_strain):
    minima = argrelmin(shear_strain)[0]
    if len(minima) < 2:
        return []
    return [(minima[i], minima[i + 1]) for i in range(min(len(minima) - 1, 4))]

def compute_secant_modulus(shear_strain, stress, bounds):
    gsec = []
    for start, end in bounds:
        ss = shear_strain[start:end+1]
        st = stress[start:end+1]
        ss_min, ss_max = np.min(ss), np.max(ss)
        st_min = np.mean(st[ss == ss_min])
        st_max = np.mean(st[ss == ss_max])
        if ss_max != ss_min:
            g = (st_max - st_min) / (ss_max - ss_min)
        else:
            g = 0.0
        gsec.append(g)
    return gsec

def compute_damping_ratio(shear_strain, stress, bounds):
    damping = []
    for start, end in bounds:
        ss = shear_strain[start:end+1]
        st = stress[start:end+1]
        ss_min, ss_max = np.min(ss), np.max(ss)
        st_min = np.mean(st[ss == ss_min])
        st_max = np.mean(st[ss == ss_max])
        g = (st_max - st_min) / (ss_max - ss_min) if ss_max != ss_min else 0
        area = trapezoid(st, ss)
        d = area / (4 * np.pi * 0.5 * (st_max - st_min) * (ss_max - ss_min)) if g != 0 else 0
        damping.append(d)
    return damping

def extract_test_no(filename):
    match = re.search(r'NO(\d+)', filename)
    return int(match.group(1)) if match else 999

def run_and_write_summary():
    files = [f for f in os.listdir('.') if f.startswith('cyclic_') and f.endswith('.txt')]

    regular = sorted([f for f in files if 'MS' not in f], key=extract_test_no)
    ms = sorted([f for f in files if 'MS' in f], key=extract_test_no)

    with open('Gsec_D_Summary.txt', 'w') as out:
        out.write("File\tGsec_1\tGsec_2\tGsec_3\tGsec_4\tD_1\tD_2\tD_3\tD_4\n")
        for category in [regular, ms]:
            for file in category:
                try:
                    cols = load_cyclic_stress_data(file)
                    shear = cols['shear_strain']
                    stress = cols['s_xz']
                    bounds = detect_cycles_by_minima(shear)
                    if len(bounds) < 4:
                        raise RuntimeError("Too few cycles")
                    gsec = compute_secant_modulus(shear, stress, bounds)
                    damp = compute_damping_ratio(shear, stress, bounds)
                    g_str = "\t".join(f"{g:.3e}" for g in gsec)
                    d_str = "\t".join(f"{d:.4f}" for d in damp)
                    out.write(f"{file}\t{g_str}\t{d_str}\n")
                except Exception as e:
                    out.write(f"{file}\tERROR: {str(e)}\n")
            out.write("\n")

if __name__ == "__main__":
    run_and_write_summary()
    print("[INFO] Summary written to Gsec_D_Summary.txt")