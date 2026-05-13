# CDSS_summary.py

import os
import re
import numpy as np

from ak_MS import (
    load_cyclic_stress_data,
    detect_cycles_by_minima,
    compute_secant_modulus,
    compute_damping_ratio
)


# strain 추출
def extract_strain(fname):
    """
    Example:
    CDSS.Shibuya1990_1E-6_2.0_2_0.7_0.5_35E9
    """
    match = re.search(r'Shibuya1990_([0-9\.E\-\+]+)_', fname)
    return float(match.group(1)) if match else None


# parameter 추출
def extract_params(fname):
    """
    Example:
    CDSS.Shibuya1990_1E-6_2.0_2_0.7_0.5_35E9

    Extract:
    RF0.7_Fric0.5_yMod35E9
    """
    match = re.search(r'_(\d+\.\d+)_(\d+\.\d+)_([0-9\.E\+\-]+)$', fname)

    if match:
        return f"RF{match.group(1)}_Fric{match.group(2)}_yMod{match.group(3)}"

    return "Unknown"


def process_file(filename):
    try:
        _, cols = load_cyclic_stress_data(filename)

        gamma = cols['shear_strain']
        tau = cols['s_xz']

        bounds = detect_cycles_by_minima(gamma)
        bounds = bounds[:4]  # first 4 cycles

        gsecs = []
        ds = []

        for start, end in bounds:
            g = gamma[start:end + 1]
            t = tau[start:end + 1]

            gsec, _, _ = compute_secant_modulus(g, t)
            damping = compute_damping_ratio(g, t)

            gsecs.append(gsec)
            ds.append(damping)

        # NaN padding if less than 4 cycles
        while len(gsecs) < 4:
            gsecs.append(np.nan)
            ds.append(np.nan)

        return gsecs, ds

    except Exception as e:
        print(f"[ERROR] {filename}: {e}")
        return [np.nan] * 4, [np.nan] * 4


def main():

    # Only CDSS files.
    # This excludes MSCDSS files in the same directory.
    files = [
        f for f in os.listdir('.')
        if f.startswith("CDSS.")
    ]

    if not files:
        print("[ERROR] No CDSS files found")
        return

    param_tag = extract_params(files[0])
    print(f"[INFO] Parameter set: {param_tag}")

    results = []

    for f in files:
        strain = extract_strain(f)

        if strain is None:
            print(f"[WARNING] Could not extract strain from {f}")
            continue

        print(f"[INFO] Processing {f}")

        gsecs, ds = process_file(f)

        results.append((strain, gsecs, ds))

    if not results:
        print("[ERROR] No valid CDSS results")
        return

    # Sort by strain
    results.sort(key=lambda x: x[0])

    # strain -> %
    strain = np.array([r[0] for r in results]) * 100.0

    # secant shear modulus
    G = np.array([r[1] for r in results])  # shape = (N, 4)

    # damping -> %
    D = np.array([r[2] for r in results]) * 100.0

    # Gmax for each cycle
    Gmax = np.nanmax(G, axis=0)

    # normalized modulus
    G_ratio = G / Gmax

    # Output filename
    outname = f"Gsec_D_cycle4_CDSS_{param_tag}.txt"

    with open(outname, "w") as f:
        f.write(
            "strain(%)\t"
            "G1\tG2\tG3\tG4\t"
            "G/Gmax1\tG/Gmax2\tG/Gmax3\tG/Gmax4\t"
            "D1(%)\tD2(%)\tD3(%)\tD4(%)\n"
        )

        for i in range(len(strain)):
            row = (
                [strain[i]] +
                list(G[i]) +
                list(G_ratio[i]) +
                list(D[i])
            )

            f.write(
                "\t".join(
                    f"{x:.4e}" if not np.isnan(x) else "NaN"
                    for x in row
                ) + "\n"
            )

    print(f"[DONE] Saved {outname}")


if __name__ == "__main__":
    main()
