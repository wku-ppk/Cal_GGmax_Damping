# CDSS_summary.py

import os
import re
import csv
import numpy as np

from ak_MS import (
    detect_cycles_by_minima,
    compute_secant_modulus,
    compute_damping_ratio
)


def load_cdss_data(filename):
    rows = []

    with open(filename, "r") as f:
        reader = csv.reader(f)
        header = None

        for row in reader:
            if not row:
                continue
            if row[0].strip().startswith("#"):
                continue

            if header is None:
                header = [h.strip() for h in row]
                continue

            rows.append([float(x.strip()) for x in row])

    if header is None:
        raise ValueError(f"No header found in {filename}")
    if len(rows) == 0:
        raise ValueError(f"No data rows found in {filename}")

    data = np.array(rows, dtype=float)
    cols = {name: data[:, i] for i, name in enumerate(header)}

    for required in ["shear_strain", "s_xz", "topcap_SXZ"]:
        if required not in cols:
            raise KeyError(f"'{required}' column not found in {filename}")

    return data, cols


def extract_strain(fname):
    """

    Example:

    CDSS.Shibuya1990FF_1.0E-4_2.0_5_1.5_0.2_30E9

    Extract:

    1.0E-4

    """

    match = re.search(r'Shibuya1990[^_]*_([0-9\.E\-\+]+)_', fname)

    return float(match.group(1)) if match else None


def extract_params(fname):
    match = re.search(r'_(\d+\.\d+)_(\d+\.\d+)_([0-9\.E\+\-]+)$', fname)
    if match:
        return f"RF{match.group(1)}_Fric{match.group(2)}_yMod{match.group(3)}"
    return "Unknown"


def calculate_cycles(gamma, tau):
    bounds = detect_cycles_by_minima(gamma)
    bounds = bounds[:4]

    gsecs = []
    ds = []

    for start, end in bounds:
        g = gamma[start:end + 1]
        t = tau[start:end + 1]

        gsec, _, _ = compute_secant_modulus(g, t)
        damping = compute_damping_ratio(g, t)

        gsecs.append(gsec)
        ds.append(damping)

    while len(gsecs) < 4:
        gsecs.append(np.nan)
        ds.append(np.nan)

    return gsecs, ds


def process_file(filename):
    try:
        _, cols = load_cdss_data(filename)

        gamma = cols["shear_strain"]

        # Original internal stress
        gsecs, ds = calculate_cycles(gamma, cols["s_xz"])

        # Top-cap stress
        gsecs_T, ds_T = calculate_cycles(gamma, cols["topcap_SXZ"])

        return gsecs, ds, gsecs_T, ds_T

    except Exception as e:
        print(f"[ERROR] {filename}: {e}")
        return [np.nan] * 4, [np.nan] * 4, [np.nan] * 4, [np.nan] * 4


def main():
    files = [
        f for f in os.listdir(".")
        if f.startswith("CDSS.")
    ]

    if not files:
        print("[ERROR] No CDSS files found")
        return

    files.sort()

    param_tag = extract_params(files[0])
    print(f"[INFO] Parameter set: {param_tag}")

    results = []

    for f in files:
        strain = extract_strain(f)

        if strain is None:
            print(f"[WARNING] Could not extract strain from {f}")
            continue

        print(f"[INFO] Processing {f}")

        gsecs, ds, gsecs_T, ds_T = process_file(f)

        results.append((strain, gsecs, ds, gsecs_T, ds_T))

    if not results:
        print("[ERROR] No valid CDSS results")
        return

    results.sort(key=lambda x: x[0])

    strain = np.array([r[0] for r in results]) * 100.0

    G = np.array([r[1] for r in results])
    D = np.array([r[2] for r in results]) * 100.0

    G_T = np.array([r[3] for r in results])
    D_T = np.array([r[4] for r in results]) * 100.0

    Gmax = np.nanmax(G, axis=0)
    G_ratio = G / Gmax

    Gmax_T = np.nanmax(G_T, axis=0)
    G_ratio_T = G_T / Gmax_T

    outname = f"Gsec_D_cycle4_CDSS_{param_tag}_StressAtTopCap.txt"

    with open(outname, "w") as f:
        f.write(
            "strain(%)\t"
            "G1\tG2\tG3\tG4\t"
            "G/Gmax1\tG/Gmax2\tG/Gmax3\tG/Gmax4\t"
            "D1(%)\tD2(%)\tD3(%)\tD4(%)\t"
            "G1_T\tG2_T\tG3_T\tG4_T\t"
            "G/Gmax1_T\tG/Gmax2_T\tG/Gmax3_T\tG/Gmax4_T\t"
            "D1(%)_T\tD2(%)_T\tD3(%)_T\tD4(%)_T\n"
        )

        for i in range(len(strain)):
            row = (
                [strain[i]] +
                list(G[i]) +
                list(G_ratio[i]) +
                list(D[i]) +
                list(G_T[i]) +
                list(G_ratio_T[i]) +
                list(D_T[i])
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
