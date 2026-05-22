#!/usr/bin/env python3
"""
plot_flight_log.py — Plot a flight log CSV produced by sim_px4.py --record

Usage:
    python3 examples/plot_flight_log.py examples/results/flight_log_YYYYMMDD_HHMMSS.csv
    python3 examples/plot_flight_log.py              # auto-selects the most recent log

Output:
    A single figure with several subplots saved next to the CSV as <log_name>.png
    and displayed interactively (close the window to exit).
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (side-effect import)

matplotlib.rcParams.update({
    "figure.dpi": 120,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "lines.linewidth": 1.2,
})

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _latest_log(results_dir: str) -> str:
    pattern = os.path.join(results_dir, "flight_log_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No flight log found in {results_dir}")
    return files[-1]


def _quat_to_euler(qx, qy, qz, qw):
    """Convert xyzw quaternion arrays to roll/pitch/yaw (rad)."""
    # roll (x-axis)
    sinr = 2.0 * (qw * qx + qy * qz)
    cosr = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr, cosr)
    # pitch (y-axis)
    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    # yaw (z-axis)
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny, cosy)
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


def _ekf_quat_to_euler(q1, q2, q3, q4):
    """MAVLink ATTITUDE_QUATERNION: q1=w, q2=x, q3=y, q4=z → roll/pitch/yaw (deg)."""
    return _quat_to_euler(q2, q3, q4, q1)


# ---------------------------------------------------------------------------
# main plotting routine
# ---------------------------------------------------------------------------

def plot(csv_path: str):
    print(f"Loading {csv_path} …")
    df = pd.read_csv(csv_path)
    t  = df["sim_time_s"].to_numpy()

    # ── derived quantities ──────────────────────────────────────────────────
    gt_roll, gt_pitch, gt_yaw = _quat_to_euler(
        df["gt_qx"].to_numpy(), df["gt_qy"].to_numpy(),
        df["gt_qz"].to_numpy(), df["gt_qw"].to_numpy(),
    )

    have_ekf = df["ekf_pos_n_m"].notna().any()
    if have_ekf:
        ekf_roll, ekf_pitch, ekf_yaw = _ekf_quat_to_euler(
            df["ekf_q1"].to_numpy(), df["ekf_q2"].to_numpy(),
            df["ekf_q3"].to_numpy(), df["ekf_q4"].to_numpy(),
        )
        # EKF NED → ENU for fair comparison with GT (x=E, y=N, z=U)
        ekf_pos_e = df["ekf_pos_e_m"].to_numpy()
        ekf_pos_n = df["ekf_pos_n_m"].to_numpy()
        ekf_pos_u = -df["ekf_pos_d_m"].to_numpy()
        ekf_vel_e = df["ekf_vel_e_ms"].to_numpy()
        ekf_vel_n = df["ekf_vel_n_ms"].to_numpy()
        ekf_vel_u = -df["ekf_vel_d_ms"].to_numpy()

    # spoofing window (for shading)
    spoofing = df["spoofing_active"].to_numpy().astype(bool)

    # ── figure layout ───────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 26))
    fig.suptitle(
        f"Flight Log — {os.path.basename(csv_path)}",
        fontsize=13, fontweight="bold", y=0.995,
    )
    gs = gridspec.GridSpec(
        6, 3,
        figure=fig,
        hspace=0.52, wspace=0.35,
        left=0.07, right=0.97, top=0.975, bottom=0.03,
    )

    def shade_spoofing(ax):
        """Shade the time axis where spoofing is active."""
        if not spoofing.any():
            return
        in_spoof = False
        t_start  = None
        for i, s in enumerate(spoofing):
            if s and not in_spoof:
                t_start  = t[i]
                in_spoof = True
            elif not s and in_spoof:
                ax.axvspan(t_start, t[i], color="red", alpha=0.10, label="_spoof")
                in_spoof = False
        if in_spoof:
            ax.axvspan(t_start, t[-1], color="red", alpha=0.10)

    # ── row 0 : 3-D trajectory ──────────────────────────────────────────────
    ax3d = fig.add_subplot(gs[0, :], projection="3d")
    ax3d.plot(df["gt_pos_x_m"], df["gt_pos_y_m"], df["gt_pos_z_m"],
              label="GT (ENU)", color="steelblue", lw=1.4)
    if have_ekf:
        ax3d.plot(ekf_pos_e, ekf_pos_n, ekf_pos_u,
                  label="EKF2 (ENU)", color="tomato", lw=1.0, alpha=0.8)
    ax3d.set_xlabel("East (m)"); ax3d.set_ylabel("North (m)"); ax3d.set_zlabel("Up (m)")
    ax3d.set_title("3-D Trajectory"); ax3d.legend(fontsize=8)

    # ── row 1 : position (ENU) ──────────────────────────────────────────────
    labels_pos = [("East (gt_pos_x_m)", "East"),
                  ("North (gt_pos_y_m)", "North"),
                  ("Up (gt_pos_z_m)", "Up")]
    gt_pos_cols  = ["gt_pos_x_m", "gt_pos_y_m", "gt_pos_z_m"]
    ekf_pos_data = [ekf_pos_e, ekf_pos_n, ekf_pos_u] if have_ekf else [None]*3

    for col, (_, lbl), ekf_data in zip(gt_pos_cols, labels_pos, ekf_pos_data):
        j = gt_pos_cols.index(col)
        ax = fig.add_subplot(gs[1, j])
        ax.plot(t, df[col].to_numpy(), label="GT", color="steelblue")
        if ekf_data is not None:
            ax.plot(t, ekf_data, label="EKF2", color="tomato", alpha=0.8)
        shade_spoofing(ax)
        ax.set_title(f"Position — {lbl}"); ax.set_xlabel("sim time (s)"); ax.set_ylabel("m")
        ax.legend(fontsize=7)

    # ── row 2 : velocity (ENU) ──────────────────────────────────────────────
    gt_vel_cols  = ["gt_vel_x_ms", "gt_vel_y_ms", "gt_vel_z_ms"]
    vel_lbls     = ["East", "North", "Up"]
    ekf_vel_data = [ekf_vel_e, ekf_vel_n, ekf_vel_u] if have_ekf else [None]*3

    for j, (col, lbl, ekf_data) in enumerate(zip(gt_vel_cols, vel_lbls, ekf_vel_data)):
        ax = fig.add_subplot(gs[2, j])
        ax.plot(t, df[col].to_numpy(), label="GT", color="steelblue")
        if ekf_data is not None:
            ax.plot(t, ekf_data, label="EKF2", color="tomato", alpha=0.8)
        shade_spoofing(ax)
        ax.set_title(f"Velocity — {lbl}"); ax.set_xlabel("sim time (s)"); ax.set_ylabel("m/s")
        ax.legend(fontsize=7)

    # ── row 3 : attitude (roll / pitch / yaw) ──────────────────────────────
    att_lbls = ["Roll", "Pitch", "Yaw"]
    gt_att   = [gt_roll,  gt_pitch,  gt_yaw]
    ekf_att  = [ekf_roll, ekf_pitch, ekf_yaw] if have_ekf else [None]*3

    for j, (lbl, gt_a, ekf_a) in enumerate(zip(att_lbls, gt_att, ekf_att)):
        ax = fig.add_subplot(gs[3, j])
        ax.plot(t, gt_a, label="GT", color="steelblue")
        if ekf_a is not None:
            ax.plot(t, ekf_a, label="EKF2", color="tomato", alpha=0.8)
        shade_spoofing(ax)
        ax.set_title(f"Attitude — {lbl}"); ax.set_xlabel("sim time (s)"); ax.set_ylabel("deg")
        ax.legend(fontsize=7)

    # ── row 4 : rotor speeds + IMU ──────────────────────────────────────────
    ax_rot = fig.add_subplot(gs[4, 0])
    for i in range(4):
        ax_rot.plot(t, df[f"rotor{i}_rads"].to_numpy(), label=f"R{i}")
    shade_spoofing(ax_rot)
    ax_rot.set_title("Rotor speeds"); ax_rot.set_xlabel("sim time (s)"); ax_rot.set_ylabel("rad/s")
    ax_rot.legend(fontsize=7)

    ax_acc = fig.add_subplot(gs[4, 1])
    for ax_col, lbl in [("imu_ax_ms2","X"),("imu_ay_ms2","Y"),("imu_az_ms2","Z")]:
        ax_acc.plot(t, df[ax_col].to_numpy(), label=lbl)
    shade_spoofing(ax_acc)
    ax_acc.set_title("IMU accelerometer"); ax_acc.set_xlabel("sim time (s)"); ax_acc.set_ylabel("m/s²")
    ax_acc.legend(fontsize=7)

    ax_gyr = fig.add_subplot(gs[4, 2])
    for ax_col, lbl in [("imu_gx_rads","X"),("imu_gy_rads","Y"),("imu_gz_rads","Z")]:
        ax_gyr.plot(t, df[ax_col].to_numpy(), label=lbl)
    shade_spoofing(ax_gyr)
    ax_gyr.set_title("IMU gyroscope"); ax_gyr.set_xlabel("sim time (s)"); ax_gyr.set_ylabel("rad/s")
    ax_gyr.legend(fontsize=7)

    # ── row 5 : EKF2 innovation ratios + GPS spoofing bias ─────────────────
    ax_innov = fig.add_subplot(gs[5, 0])
    if have_ekf:
        for col, lbl in [
            ("ekf_vel_ratio",       "vel"),
            ("ekf_pos_horiz_ratio", "pos_h"),
            ("ekf_pos_vert_ratio",  "pos_v"),
            ("ekf_mag_ratio",       "mag"),
        ]:
            ax_innov.plot(t, df[col].to_numpy(), label=lbl)
        ax_innov.axhline(1.0, color="red", lw=0.8, ls="--", label="reject threshold")
    shade_spoofing(ax_innov)
    ax_innov.set_title("EKF2 innovation ratios (>1 = rejected)")
    ax_innov.set_xlabel("sim time (s)"); ax_innov.set_ylabel("ratio")
    ax_innov.legend(fontsize=7)

    ax_acc_est = fig.add_subplot(gs[5, 1])
    if have_ekf:
        ax_acc_est.plot(t, df["ekf_pos_horiz_acc_m"].to_numpy(), label="horiz σ")
        ax_acc_est.plot(t, df["ekf_pos_vert_acc_m"].to_numpy(),  label="vert σ")
    shade_spoofing(ax_acc_est)
    ax_acc_est.set_title("EKF2 position accuracy (1-σ)")
    ax_acc_est.set_xlabel("sim time (s)"); ax_acc_est.set_ylabel("m")
    ax_acc_est.legend(fontsize=7)

    ax_spoof = fig.add_subplot(gs[5, 2])
    ax_spoof.plot(t, df["spoof_bias_x_m"].to_numpy(), label="X (E)")
    ax_spoof.plot(t, df["spoof_bias_y_m"].to_numpy(), label="Y (N)")
    ax_spoof.plot(t, df["spoof_bias_z_m"].to_numpy(), label="Z (U)")
    shade_spoofing(ax_spoof)
    ax_spoof.set_title("GPS spoof bias (ENU)"); ax_spoof.set_xlabel("sim time (s)"); ax_spoof.set_ylabel("m")
    ax_spoof.legend(fontsize=7)

    # ── save ────────────────────────────────────────────────────────────────
    out_png = os.path.splitext(csv_path)[0] + ".png"
    fig.savefig(out_png, bbox_inches="tight")
    print(f"Saved → {out_png}")
    plt.show()


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot a sim_px4.py flight log CSV")
    parser.add_argument(
        "log_file", nargs="?", default=None,
        help="Path to flight_log_*.csv. Omit to use the most recent log in examples/results/",
    )
    args, _ = parser.parse_known_args()  # use parse_known_args to ignore extra args from IDE

    if args.log_file:
        csv_path = args.log_file
    else:
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        csv_path = _latest_log(results_dir)
        print(f"Auto-selected: {csv_path}")

    if not os.path.isfile(csv_path):
        sys.exit(f"File not found: {csv_path}")

    plot(csv_path)


if __name__ == "__main__":
    main()
