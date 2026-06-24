# ============================================================
# Script 09: Prepare report-ready robustness tables and figures
# Project: Robustness of Medical Image Segmentation Models
# ============================================================

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

PROJECT_ROOT = Path("/home/xfh25/brats_segmentation_project")

INPUT_CSV = PROJECT_ROOT / "results/08_full_degraded_test_summary.csv"

REPORT_DIR = PROJECT_ROOT / "report_materials"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_TABLE_CSV = REPORT_DIR / "09_robustness_report_table.csv"
REPORT_TABLE_TXT = REPORT_DIR / "09_robustness_report_table.txt"

DICE_CURVE_PNG = REPORT_DIR / "09_robustness_mean_tumor_dice_curve.png"
LEVEL5_DROP_PNG = REPORT_DIR / "09_level5_dice_drop_bar.png"


# ------------------------------------------------------------
# 2. Load full robustness summary
# ------------------------------------------------------------

df = pd.read_csv(INPUT_CSV)

print("=" * 80)
print("Script 09: Prepare report-ready robustness materials")
print("=" * 80)

print(f"Loaded: {INPUT_CSV}")
print(f"Rows: {len(df)}")


# ------------------------------------------------------------
# 3. Make clean report table
# ------------------------------------------------------------

report_df = df[
    [
        "degradation_type",
        "severity_level",
        "severity_value",
        "mean_tumor_dice_tumor_slices_only",
        "mean_tumor_iou_tumor_slices_only",
        "dice_drop_from_clean",
        "iou_drop_from_clean",
    ]
].copy()

report_df = report_df.rename(
    columns={
        "degradation_type": "Degradation",
        "severity_level": "Severity Level",
        "severity_value": "Severity Value",
        "mean_tumor_dice_tumor_slices_only": "Mean Tumor Dice",
        "mean_tumor_iou_tumor_slices_only": "Mean Tumor IoU",
        "dice_drop_from_clean": "Dice Drop from Clean",
        "iou_drop_from_clean": "IoU Drop from Clean",
    }
)

# Round numeric values for cleaner reporting
for col in [
    "Mean Tumor Dice",
    "Mean Tumor IoU",
    "Dice Drop from Clean",
    "IoU Drop from Clean",
]:
    report_df[col] = report_df[col].round(4)

report_df.to_csv(REPORT_TABLE_CSV, index=False)

with open(REPORT_TABLE_TXT, "w") as f:
    f.write(report_df.to_string(index=False))

print(f"Saved clean CSV table: {REPORT_TABLE_CSV}")
print(f"Saved readable TXT table: {REPORT_TABLE_TXT}")


# ------------------------------------------------------------
# 4. Plot Dice curve across severity levels
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

for degradation in report_df["Degradation"].unique():
    temp = report_df[report_df["Degradation"] == degradation]
    temp = temp.sort_values("Severity Level")

    plt.plot(
        temp["Severity Level"],
        temp["Mean Tumor Dice"],
        marker="o",
        label=degradation
    )

plt.axhline(
    y=0.3917,
    linestyle="--",
    linewidth=1,
    label="Clean reference"
)

plt.xlabel("Severity Level")
plt.ylabel("Mean Tumor Dice")
plt.title("2D U-Net Robustness Under Image Quality Degradation")
plt.xticks([1, 2, 3, 4, 5])
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(DICE_CURVE_PNG, dpi=300)
plt.close()

print(f"Saved Dice curve figure: {DICE_CURVE_PNG}")


# ------------------------------------------------------------
# 5. Plot level-5 Dice drop from clean
# ------------------------------------------------------------

level5_df = report_df[report_df["Severity Level"] == 5].copy()
level5_df = level5_df.sort_values("Dice Drop from Clean", ascending=False)

plt.figure(figsize=(8, 5))

plt.bar(
    level5_df["Degradation"],
    level5_df["Dice Drop from Clean"]
)

plt.xlabel("Degradation Type")
plt.ylabel("Dice Drop from Clean")
plt.title("Performance Drop at Highest Degradation Severity")
plt.xticks(rotation=30, ha="right")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig(LEVEL5_DROP_PNG, dpi=300)
plt.close()

print(f"Saved level-5 drop figure: {LEVEL5_DROP_PNG}")


# ------------------------------------------------------------
# 6. Print short interpretation
# ------------------------------------------------------------

worst_row = level5_df.iloc[0]
best_row = level5_df.iloc[-1]

print("-" * 80)
print("Level-5 degradation ranking:")
print(level5_df[["Degradation", "Mean Tumor Dice", "Dice Drop from Clean"]].to_string(index=False))

print("-" * 80)
print(
    f"Most damaging degradation at level 5: "
    f"{worst_row['Degradation']} "
    f"(Dice drop = {worst_row['Dice Drop from Clean']:.4f})"
)

print(
    f"Least damaging degradation at level 5: "
    f"{best_row['Degradation']} "
    f"(Dice drop = {best_row['Dice Drop from Clean']:.4f})"
)

print("=" * 80)
print("Script 09 finished.")