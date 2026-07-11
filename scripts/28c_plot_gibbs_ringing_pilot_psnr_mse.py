#!/usr/bin/env python3
"""
Script 28C: Plot MSE and PSNR curves for Gibbs-like ringing pilot.

Purpose:
- Read Script 28B CSV.
- Create report-ready MSE and PSNR plots.
- Save small plot files in report_materials/.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")

IN_CSV = PROJECT_ROOT / "report_materials/28b_gibbs_ringing_pilot_psnr_mse.csv"

OUT_MSE = PROJECT_ROOT / "report_materials/28b_gibbs_ringing_pilot_mse_curve.png"
OUT_PSNR = PROJECT_ROOT / "report_materials/28b_gibbs_ringing_pilot_psnr_curve.png"


def main():
    df = pd.read_csv(IN_CSV)

    # Extract numeric level from strings like gibbs_L1
    df["level_num"] = df["level"].str.extract(r"L(\d+)").astype(int)
    df = df.sort_values("level_num")

    # MSE plot
    plt.figure(figsize=(7, 5))
    plt.plot(df["level_num"], df["mse"], marker="o")
    plt.xlabel("Gibbs ringing severity level")
    plt.ylabel("MSE")
    plt.title("Gibbs-like ringing pilot: MSE by severity")
    plt.xticks(df["level_num"])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_MSE, dpi=200)
    plt.close()

    # PSNR plot
    plt.figure(figsize=(7, 5))
    plt.plot(df["level_num"], df["psnr_db"], marker="o")
    plt.xlabel("Gibbs ringing severity level")
    plt.ylabel("PSNR (dB)")
    plt.title("Gibbs-like ringing pilot: PSNR by severity")
    plt.xticks(df["level_num"])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PSNR, dpi=200)
    plt.close()

    print("=" * 80)
    print("Script 28C: Gibbs-like ringing pilot MSE/PSNR plots")
    print("=" * 80)
    print(f"Read:      {IN_CSV}")
    print(f"Saved MSE: {OUT_MSE}")
    print(f"Saved PSNR:{OUT_PSNR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
