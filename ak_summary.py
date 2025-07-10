import os
import sys
import numpy as np
from ak import load_cyclic_stress_data, detect_cycles_by_minima, compute_secant_modulus, compute_damping_ratio
import re

def extract_number(fname):
    match = re.search(r'NO(\d+)', fname)
    return int(match.group(1)) if match else float('inf')

def summarize_file(filename):
    try:
        _, cols = load_cyclic_stress_data(filename)
        shear_strain = cols['shear_strain']
        s_xz = cols['s_xz']
        bounds = detect_cycles_by_minima(shear_strain)
        bounds = bounds[:4]  # max 4 cycles

        gsecs = []
        ds = []
        for start, end in bounds:
            gamma = shear_strain[start:end + 1]
            tau = s_xz[start:end + 1]
            gsec, _, _ = compute_secant_modulus(gamma, tau)
            damping = compute_damping_ratio(gamma, tau)
            gsecs.append(gsec)
            ds.append(damping)

        while len(gsecs) < 4:
            gsecs.append(np.nan)
            ds.append(np.nan)

        return [filename] + gsecs + ds

    except Exception as e:
        print(f"[ERROR] {filename}: {e}")
        return [filename] + [np.nan]*8

def main():
    files = sorted(
        [f for f in os.listdir('.') if f.startswith("cyclic_stress_NO") and f.endswith(".txt")],
        key=extract_number
    )
    results = []
    for f in files:
        print(f"[INFO] Processing {f}")
        results.append(summarize_file(f))

    with open("Gsec_D_Summary.txt", "w") as out:
        header = "File\tGsec_1\tGsec_2\tGsec_3\tGsec_4\tD_1\tD_2\tD_3\tD_4\n"
        out.write(header)
        for row in results:
            line = row[0] + "\t" + "\t".join(f"{x:.3e}" if not np.isnan(x) else "NaN" for x in row[1:]) + "\n"
            out.write(line)

    print("[DONE] Results saved to Gsec_D_Summary.txt")

if __name__ == "__main__":
    main()