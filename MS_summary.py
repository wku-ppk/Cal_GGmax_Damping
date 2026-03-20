import os
import re
import numpy as np

from ak_MS import (
    load_cyclic_stress_data,
    detect_cycles_by_minima,
    compute_secant_modulus,
    compute_damping_ratio
)

# 🔥 strain 추출
def extract_strain(fname):
    match = re.search(r'Shibuya1990_([0-9\.E\-\+]+)_', fname)
    return float(match.group(1)) if match else None

# 🔥 parameter 추출 (출력 파일명용)
def extract_params(fname):
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
        bounds = bounds[:4]  # 첫 4개 cycle

        gsecs = []
        ds = []

        for start, end in bounds:
            g = gamma[start:end+1]
            t = tau[start:end+1]

            gsec, _, _ = compute_secant_modulus(g, t)
            damping = compute_damping_ratio(g, t)

            gsecs.append(gsec)
            ds.append(damping)

        # 부족하면 NaN padding
        while len(gsecs) < 4:
            gsecs.append(np.nan)
            ds.append(np.nan)

        return gsecs, ds

    except Exception as e:
        print(f"[ERROR] {filename}: {e}")
        return [np.nan]*4, [np.nan]*4


def main():

    files = [f for f in os.listdir('.') if f.startswith("MSCDSS")]

    if not files:
        print("[ERROR] No files found")
        return

    param_tag = extract_params(files[0])
    print(f"[INFO] Parameter set: {param_tag}")

    results = []

    for f in files:
        strain = extract_strain(f)
        if strain is None:
            continue

        print(f"[INFO] Processing {f}")

        gsecs, ds = process_file(f)

        results.append((strain, gsecs, ds))

    # strain 기준 정렬
    results.sort(key=lambda x: x[0])

    strain = np.array([r[0] for r in results])

    G = np.array([r[1] for r in results])  # (N,4)
    D = np.array([r[2] for r in results])  # (N,4)

    # 🔥 각 cycle별 Gmax 계산
    Gmax = np.nanmax(G, axis=0)  # cycle별 max

    G_ratio = G / Gmax  # broadcasting

    # 🔥 출력 파일명
    outname = f"Gsec_D_cycle4_{param_tag}.txt"

    with open(outname, "w") as f:
        f.write("strain\tG1\tG2\tG3\tG4\tG/Gmax1\tG/Gmax2\tG/Gmax3\tG/Gmax4\tD1\tD2\tD3\tD4\n")

        for i in range(len(strain)):
            row = [strain[i]] + \
                  list(G[i]) + \
                  list(G_ratio[i]) + \
                  list(D[i])

            f.write("\t".join(
                f"{x:.4e}" if not np.isnan(x) else "NaN"
                for x in row
            ) + "\n")

    print(f"[DONE] Saved {outname}")


if __name__ == "__main__":
    main()
