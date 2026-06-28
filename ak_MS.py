import os
import sys
import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelmin


def load_cyclic_stress_data(filename):
    """
    Load an MSCDSS measurement-sphere stress file.

    Expected numeric columns:
        step, time, shear_strain,
        s_xx, s_yy, s_zz,
        s_xy, s_xz, s_yz
    """

    if not os.path.isfile(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    def find_data_start(lines):
        for i, line in enumerate(lines):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                values = [
                    float(x.strip())
                    for x in stripped.split(",")
                ]

                if values:
                    return i

            except ValueError:
                continue

        raise ValueError(
            f"No numeric data found in '{filename}'."
        )

    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    start_row = find_data_start(lines)

    try:
        data = np.loadtxt(
            filename,
            delimiter=",",
            skiprows=start_row,
            ndmin=2,
        )

    except Exception as error:
        raise ValueError(
            f"Failed to load numeric data from "
            f"'{filename}': {error}"
        ) from error

    column_names = [
        "step",
        "time",
        "shear_strain",
        "s_xx",
        "s_yy",
        "s_zz",
        "s_xy",
        "s_xz",
        "s_yz",
    ]

    if data.shape[1] != len(column_names):
        raise ValueError(
            f"Column count mismatch in '{filename}': "
            f"expected {len(column_names)}, "
            f"but found {data.shape[1]}."
        )

    columns = {
        name: data[:, i]
        for i, name in enumerate(column_names)
    }

    print(
        f"[INFO] Loaded {data.shape[0]} rows from "
        f"'{filename}' "
        f"(data starts at line {start_row + 1})"
    )

    return data, columns


def detect_cycles_by_minima(shear_strain, order=1):
    """
    Detect complete cycles using successive local minima of shear strain.

    A cycle is defined from one negative strain peak to the next
    negative strain peak.

    Parameters
    ----------
    shear_strain : array-like
        Shear strain history.

    order : int
        Number of neighboring points used by argrelmin when identifying
        each local minimum.

    Returns
    -------
    list of tuple
        List of (start_index, end_index) cycle boundaries.
    """

    gamma = np.asarray(shear_strain, dtype=float)

    if len(gamma) < 3:
        return []

    valid = np.isfinite(gamma)

    if not np.all(valid):
        raise ValueError(
            "Shear-strain data contain NaN or infinite values."
        )

    minima_indices = argrelmin(
        gamma,
        order=order,
    )[0]

    if minima_indices.size < 2:
        return []

    return [
        (
            int(minima_indices[i]),
            int(minima_indices[i + 1]),
        )
        for i in range(len(minima_indices) - 1)
    ]


def compute_secant_modulus(shear_strain, stress):
    """
    Calculate peak-to-peak secant shear modulus.

    The stress values are evaluated at the minimum and maximum
    strain points:

        Gsec = [tau(gamma_max) - tau(gamma_min)]
               / [gamma_max - gamma_min]

    A constant stress bias therefore does not affect Gsec.

    Returns
    -------
    gsec : float
        Secant shear modulus.

    point_min : tuple
        (gamma_min, stress at gamma_min)

    point_max : tuple
        (gamma_max, stress at gamma_max)
    """

    gamma = np.asarray(shear_strain, dtype=float)
    tau = np.asarray(stress, dtype=float)

    valid = np.isfinite(gamma) & np.isfinite(tau)
    gamma = gamma[valid]
    tau = tau[valid]

    if len(gamma) < 2:
        return (
            np.nan,
            (np.nan, np.nan),
            (np.nan, np.nan),
        )

    i_min = int(np.argmin(gamma))
    i_max = int(np.argmax(gamma))

    gamma_min = gamma[i_min]
    gamma_max = gamma[i_max]

    tau_at_gamma_min = tau[i_min]
    tau_at_gamma_max = tau[i_max]

    delta_gamma = gamma_max - gamma_min
    delta_tau = (
        tau_at_gamma_max
        - tau_at_gamma_min
    )

    if np.isclose(delta_gamma, 0.0):
        return (
            np.nan,
            (gamma_min, tau_at_gamma_min),
            (gamma_max, tau_at_gamma_max),
        )

    gsec = delta_tau / delta_gamma

    return (
        gsec,
        (gamma_min, tau_at_gamma_min),
        (gamma_max, tau_at_gamma_max),
    )


def _trapezoidal_integral(y, x):
    """
    Compatibility helper for different NumPy versions.
    """

    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)

    return np.trapz(y, x)


def compute_damping_ratio(
    shear_strain,
    stress,
    close_loop=True,
    closure_warning=False,
):
    """
    Calculate equivalent hysteretic damping for one complete cycle.

    The method uses:

        D = Delta_W / (4*pi*Ws)

    where:

        Delta_W = area enclosed by the hysteresis loop

        Ws = 0.5 * tau_amplitude * gamma_amplitude

    Therefore:

        D = Delta_W
            / (2*pi*tau_amplitude*gamma_amplitude)

    The strain and stress amplitudes are calculated from the stresses
    corresponding to the positive and negative strain extrema:

        gamma_amplitude =
            (gamma_max - gamma_min) / 2

        tau_amplitude =
            abs[tau(gamma_max) - tau(gamma_min)] / 2

    This formulation is suitable when the imposed strain is symmetric
    but the measured stress loop has a nonzero mean stress or stress
    bias.

    Parameters
    ----------
    shear_strain : array-like
        One complete cycle of shear-strain data.

    stress : array-like
        Corresponding shear-stress data.

    close_loop : bool
        If True, add a straight segment from the last point back to
        the first point. This compensates for a small sampling gap at
        the cycle boundary.

    closure_warning : bool
        If True, print a warning when the first and last points differ
        appreciably relative to the cycle ranges.

    Returns
    -------
    float
        Equivalent damping ratio as a decimal value.
        Multiply by 100 to express it in percent.
    """

    gamma = np.asarray(shear_strain, dtype=float)
    tau = np.asarray(stress, dtype=float)

    valid = np.isfinite(gamma) & np.isfinite(tau)
    gamma = gamma[valid]
    tau = tau[valid]

    if len(gamma) < 4:
        return np.nan

    # -------------------------------------------------------------
    # 1. Find strain extrema
    # -------------------------------------------------------------

    i_min = int(np.argmin(gamma))
    i_max = int(np.argmax(gamma))

    gamma_min = gamma[i_min]
    gamma_max = gamma[i_max]

    tau_at_gamma_min = tau[i_min]
    tau_at_gamma_max = tau[i_max]

    # -------------------------------------------------------------
    # 2. Calculate cyclic amplitudes
    # -------------------------------------------------------------

    gamma_amplitude = (
        gamma_max - gamma_min
    ) / 2.0

    tau_amplitude = abs(
        tau_at_gamma_max
        - tau_at_gamma_min
    ) / 2.0

    if (
        np.isclose(gamma_amplitude, 0.0)
        or np.isclose(tau_amplitude, 0.0)
    ):
        return np.nan

    # -------------------------------------------------------------
    # 3. Determine loop centers
    #
    # Subtracting a constant stress bias does not change the area
    # of a perfectly closed loop, but it explicitly centers the
    # calculation and makes the treatment of the bias clear.
    # -------------------------------------------------------------

    gamma_center = (
        gamma_max + gamma_min
    ) / 2.0

    tau_center = (
        tau_at_gamma_max
        + tau_at_gamma_min
    ) / 2.0

    gamma_corrected = gamma - gamma_center
    tau_corrected = tau - tau_center

    # -------------------------------------------------------------
    # 4. Check whether the detected cycle is reasonably closed
    # -------------------------------------------------------------

    gamma_range = np.ptp(gamma)
    tau_range = np.ptp(tau)

    gamma_closure_ratio = (
        abs(gamma[-1] - gamma[0])
        / gamma_range
        if gamma_range > 0.0
        else np.nan
    )

    tau_closure_ratio = (
        abs(tau[-1] - tau[0])
        / tau_range
        if tau_range > 0.0
        else np.nan
    )

    if closure_warning:
        if (
            gamma_closure_ratio > 0.02
            or tau_closure_ratio > 0.10
        ):
            print(
                "[WARNING] Hysteresis loop may not be closed: "
                f"strain closure = "
                f"{gamma_closure_ratio:.2%}, "
                f"stress closure = "
                f"{tau_closure_ratio:.2%}"
            )

    # -------------------------------------------------------------
    # 5. Integrate the recorded hysteresis path
    # -------------------------------------------------------------

    path_integral = _trapezoidal_integral(
        tau_corrected,
        gamma_corrected,
    )

    closing_integral = 0.0

    if close_loop:
        closing_integral = (
            0.5
            * (
                tau_corrected[-1]
                + tau_corrected[0]
            )
            * (
                gamma_corrected[0]
                - gamma_corrected[-1]
            )
        )

    dissipated_energy = abs(
        path_integral + closing_integral
    )

    # -------------------------------------------------------------
    # 6. Calculate reference strain energy and damping
    # -------------------------------------------------------------

    reference_strain_energy = (
        0.5
        * tau_amplitude
        * gamma_amplitude
    )

    if np.isclose(
        reference_strain_energy,
        0.0,
    ):
        return np.nan

    damping_ratio = (
        dissipated_energy
        / (
            4.0
            * np.pi
            * reference_strain_energy
        )
    )

    return damping_ratio


def plot_full_and_individual_cycles(
    shear_strain,
    s_xz,
    cycle_bounds,
    label_prefix="Cycle",
):
    """
    Plot the full stress-strain history and each detected cycle.
    """

    gamma_all = np.asarray(
        shear_strain,
        dtype=float,
    )

    tau_all = np.asarray(
        s_xz,
        dtype=float,
    )

    num_cycles = len(cycle_bounds)
    total_plots = num_cycles + 1

    cols = 3
    rows = math.ceil(total_plots / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(5 * cols, 4 * rows),
    )

    axes = np.atleast_1d(axes).flatten()

    axes[0].plot(
        gamma_all,
        tau_all,
        label="Full Data",
    )

    axes[0].set_title(
        "Full Hysteresis Curve"
    )

    axes[0].set_xlabel(
        "Shear Strain"
    )

    axes[0].set_ylabel(
        "Shear Stress, s_xz"
    )

    axes[0].grid(True)
    axes[0].legend()

    for i, (start, end) in enumerate(
        cycle_bounds
    ):
        gamma = gamma_all[start:end + 1]
        tau = tau_all[start:end + 1]

        gsec, point_min, point_max = (
            compute_secant_modulus(
                gamma,
                tau,
            )
        )

        damping = compute_damping_ratio(
            gamma,
            tau,
            close_loop=True,
            closure_warning=False,
        )

        ax = axes[i + 1]

        ax.plot(
            gamma,
            tau,
            label=f"{label_prefix} {i + 1}",
        )

        ax.plot(
            [
                point_min[0],
                point_max[0],
            ],
            [
                point_min[1],
                point_max[1],
            ],
            "r--",
            label="Secant Line",
        )

        if np.isfinite(damping):
            damping_text = (
                f"{damping * 100.0:.3f}%"
            )
        else:
            damping_text = "NaN"

        if np.isfinite(gsec):
            gsec_text = f"{gsec:.3e}"
        else:
            gsec_text = "NaN"

        ax.set_title(
            f"{label_prefix} {i + 1}\n"
            f"Gsec = {gsec_text}, "
            f"D = {damping_text}"
        )

        ax.set_xlabel(
            "Shear Strain"
        )

        ax.set_ylabel(
            "Shear Stress, s_xz"
        )

        ax.grid(True)
        ax.legend()

    for j in range(
        total_plots,
        len(axes),
    ):
        fig.delaxes(axes[j])

    fig.suptitle(
        "Full and Individual Hysteresis Cycles",
        fontsize=16,
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    plt.show()


def main():
    if len(sys.argv) != 2:
        print(
            "Usage: "
            "python3 ak_MS.py <MSCDSS_filename>"
        )
        sys.exit(1)

    filename = sys.argv[1]

    try:
        _, columns = load_cyclic_stress_data(
            filename
        )

        shear_strain = columns[
            "shear_strain"
        ]

        s_xz = columns["s_xz"]

        cycle_bounds = detect_cycles_by_minima(
            shear_strain
        )

        if not cycle_bounds:
            print(
                "[ERROR] No complete cycles "
                "were detected."
            )
            sys.exit(1)

        print(
            f"[INFO] Detected "
            f"{len(cycle_bounds)} complete cycles."
        )

        for i, (start, end) in enumerate(
            cycle_bounds,
            start=1,
        ):
            gamma = shear_strain[start:end + 1]
            tau = s_xz[start:end + 1]

            gsec, _, _ = compute_secant_modulus(
                gamma,
                tau,
            )

            damping = compute_damping_ratio(
                gamma,
                tau,
                close_loop=True,
                closure_warning=True,
            )

            print(
                f"[RESULT] Cycle {i}: "
                f"Gsec = {gsec:.6e}, "
                f"D = {damping:.6f}, "
                f"D = {damping * 100.0:.4f}%"
            )

        plot_full_and_individual_cycles(
            shear_strain,
            s_xz,
            cycle_bounds,
        )

    except Exception as error:
        print(f"[ERROR] {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()