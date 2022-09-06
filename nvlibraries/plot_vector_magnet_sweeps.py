import numpy as np
import matplotlib.pyplot as plt

# import lmfit
import pandas as pd
import h5py
import os
import sys
import pprint

import sys

sys.path.append(r"C:\Users\QMPL\Documents\GitHub\NVfunctions")
from nv_functions import fit_nv_angle, angle_dependent_splitting


def read_h5(path):
    with h5py.File(path, "r") as f:
        axis = f["odmr/frequency_axis"][:]
        trace = f["odmr/odmr_average_trace"][:]
        l0_center_0 = f["/fit/range: 0/odmr_fit"].attrs["l0_center"]
        l0_center_1 = f["/fit/range: 1/odmr_fit"].attrs["l0_center"]

    return axis, trace, l0_center_0, l0_center_1


def plot_trace(axis, trace):
    plt.plot(axis, trace, linestyle="--", color="#666666")
    plt.scatter(axis, trace, marker="x")
    plt.show()


def read_angles(df, path, key):
    angles = []
    splittings = []
    for file in os.listdir(path):
        if not file.endswith(".h5"):
            continue

        file_time = int(file.split("-")[1])
        if not file_time in df["file"].values:
            continue

        row = df[df["file"] == file_time].iloc[0]
        angles.append(row[key])

        axis, trace, l0_center, l1_center = read_h5(f"{path}/{file}")
        splittings.append(l1_center - l0_center)

    angles = np.array(angles)
    splittings = np.array(splittings)

    sorting_ind = np.argsort(angles)
    angles = angles[sorting_ind]
    splittings = splittings[sorting_ind]

    output = np.zeros((len(angles), 2))
    output[:, 0] = np.array(angles) * np.pi / 180
    output[:, 1] = np.array(splittings)
    return output


def use_fit_results(Btheta, Bphi, Esplit, fit_pars):
    NVtheta, NVphi, Bnorm = fit_pars
    return angle_dependent_splitting(
        Bnorm=Bnorm,
        Btheta=Btheta,
        Bphi=Bphi,
        NVtheta=NVtheta,
        NVphi=NVphi,
        Esplit=Esplit,
    )


path_polar = r"C:\Users\QMPL\Documents\Qudi\data\2022\07\20220729\odmr_logic"
path_azimuthal = r"C:\Users\QMPL\Documents\Qudi\data\2022\07\20220728\odmr_logic"

df_polar = pd.read_csv(f"{path_polar}/polar_sweep.txt")
df_azimuthal = pd.read_csv(f"{path_azimuthal}/azimuthal_sweep.txt")

polar_arr = read_angles(df_polar, path_polar, "polar angle [deg]")
azi_arr = read_angles(df_azimuthal, path_azimuthal, "azimuthal_angle [deg]")

fixed_theta = np.deg2rad(
    90
)  # The value for the polar angle that is kept constant during the azimuthal sweep
fixed_phi = np.deg2rad(91.4)  # Vice versa
Esplit = 3.03e6  # Measured as the zero field splitting

# fit_pars are [NV polar angle, NV azimuthal angle, B field magnitude].
fit_pars, out1, out2 = fit_nv_angle(
    phi_sweep_and_split=azi_arr,
    theta_sweep_and_split=polar_arr,
    fixed_theta=fixed_theta,
    fixed_phi=fixed_phi,
    Esplit=Esplit,
    x0=[1.0, 0, 0],
)

# Plot the fit results
NVtheta, NVphi, _ = fit_pars
print(f"Polar Angle: {np.rad2deg(NVtheta)}")
print(f"Azimuthal Angle: {np.rad2deg(NVphi)}")

fit_angles = np.linspace(0, np.pi)
azi_fit_result = use_fit_results(
    Btheta=fixed_theta, Bphi=fit_angles, Esplit=Esplit, fit_pars=fit_pars
)
fig, ax = plt.subplots()
ax.scatter(
    np.rad2deg(azi_arr[:, 0]), azi_arr[:, 1] * 1e-6, marker="x",
)
ax.plot(
    np.rad2deg(fit_angles), azi_fit_result * 1e-6, color="orange", linestyle="--",
)
ax.set_ylim(bottom=0)
ax.set_xlim(0, 180)
ax.set_ylabel("Splitting [MHz]")
ax.set_xlabel("Azimuthal Angle [deg]")
ax.set_title("Azimuthal Sweep")
plt.show()

polar_fit_result = use_fit_results(
    Btheta=fit_angles, Bphi=fixed_phi, Esplit=Esplit, fit_pars=fit_pars
)
fig, ax = plt.subplots()
ax.scatter(np.rad2deg(polar_arr[:, 0]), polar_arr[:, 1] * 1e-6, marker="x")
ax.plot(np.rad2deg(fit_angles), polar_fit_result * 1e-6, color="orange", linestyle="--")
ax.set_ylim(bottom=0)
ax.set_xlim(0, 180)
ax.set_ylabel("Splitting [MHz]")
ax.set_xlabel("Polar Angle [deg]")
ax.set_title("Polar Sweep")
plt.show()
