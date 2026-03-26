import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from datetime import datetime
from math import sqrt
import glob
from scipy.stats import binom

TARGET_cFAR = 0.3
THRESHOLDS = np.arange(0, 1.01, 0.01)  # {0,0.01,...,1.00}
CALIBRATION_RATIO = 0.2
RANDOM_SEED = 1

START_L_LIST = [1, 3, 5, 10, 15, 20] # representative choice for the paper: L=10

model_name = "gpt-5-mini"
correct = "+1"
incorrect = "-1"
abstain = "+0.4"

RESULT_DIR = f"main_exp/outputs/{model_name}/cfar_plots_mfs"
os.makedirs(RESULT_DIR, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

DELTA = 0.05
RISK_TARGETS = [0.1, 0.2, 0.3, 0.4]

def get_latest_file(folder, pattern):
    files = glob.glob(os.path.join(folder, pattern))
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    return max(files, key=os.path.getmtime)

def load_files(model_name, correct, incorrect, abstain):
    base_dir = f"main_exp/example_output/{model_name}"
    scheme_info = {
        f"Scheme A ({correct}, {incorrect})": f"popqa_A_{correct}_{incorrect}_*.csv",
        f"Scheme B ({correct}, {incorrect}, {abstain})": f"popqa_B_{correct}_{incorrect}_{abstain}_*.csv",
        f"Scheme B w/ norms ({correct}, {incorrect}, {abstain})": f"popqa_B_norm_{correct}_{incorrect}_{abstain}_*.csv"
    }
    return {scheme_name: get_latest_file(base_dir, pattern) for scheme_name, pattern in scheme_info.items()}

def wilson_ci(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return center - margin, center + margin

def ci_to_pm(ci):
    low, high = ci
    return (high - low) / 2

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    if "first_confidence" in df.columns:
        df["confidence"] = df["first_confidence"].combine_first(df["best_guess_confidence"])
    else:
        df["confidence"] = df["best_guess_confidence"]
    df["uncertainty"] = 1.0 - df["confidence"]
    return df

def split_data(df):
    if CALIBRATION_RATIO == 1.0:
        return df, pd.DataFrame()
    return train_test_split(df, train_size=CALIBRATION_RATIO, random_state=RANDOM_SEED, shuffle=True)

def compute_cfar(df_subset):
    if len(df_subset) == 0:
        return 0.0
    return (1 - df_subset["correct"]).mean()


# Algorithm 3: Multistart Fixed-Sequence CP
def multistart_fixed_sequence_threshold_selection(df_cal, target_r):
    certified_thresholds = []

    L = len(START_POINTS)
    delta_per_start = DELTA / L

    certified_by_start = {}  # track each path

    for start_idx in START_POINTS:
        j = start_idx
        path_certified = []  

        while j < len(THRESHOLDS):
            t = THRESHOLDS[j]

            accepted = df_cal[df_cal["uncertainty"] <= t]
            n = len(accepted)
            k = int((1 - accepted["correct"]).sum())

            p_val = 1.0 if n == 0 else binom.cdf(k, n, target_r)

            if p_val <= delta_per_start:
                path_certified.append(t)
                certified_thresholds.append(t)
                j += 1
            else:
                break

        certified_by_start[start_idx] = path_certified

    if len(certified_thresholds) == 0:
        return None
    else:
        return max(certified_thresholds)

def construct_empirical_far_curve_validation(df):
    cfars = [compute_cfar(df[df["uncertainty"] <= t]) for t in THRESHOLDS]
    return THRESHOLDS.copy(), np.array(cfars)

def compute_cfar_curve_with_ci(df):
    cfar_vals, ci_low, ci_high = [], [], []

    for t in THRESHOLDS:
        accepted = df[df["uncertainty"] <= t]
        n = len(accepted)
        if n == 0:
            cfar_vals.append(0.0)
            ci_low.append(0.0)
            ci_high.append(0.0)
            continue
        k = int((1 - accepted["correct"]).sum())
        cfar = k / n
        low, high = wilson_ci(k, n)
        cfar_vals.append(cfar)
        ci_low.append(low)
        ci_high.append(high)

    # force last point
    cfar_vals[-1] = compute_cfar(df)
    return THRESHOLDS.copy(), np.array(cfar_vals), np.array(ci_low), np.array(ci_high)

def validate_with_ci(val_df, threshold):
    accepted = val_df[val_df["uncertainty"] <= threshold]
    n_val = len(val_df)
    n_acc = len(accepted)
    ar_low, ar_high = wilson_ci(n_acc, n_val)
    k_false = int((1 - accepted["correct"]).sum())
    cfar = compute_cfar(accepted)
    cfar_low, cfar_high = wilson_ci(k_false, n_acc)
    return {
        "val_accept_rate": n_acc / n_val,
        "val_accept_ci": (ar_low, ar_high),
        "validation_cfar": cfar,
        "validation_cfar_ci": (cfar_low, cfar_high)
    }

def run_all_schemes():
    fig = plt.figure(figsize=(9,6))
    val_points = {}
    style_map = {}

    for scheme_name, file_path in FILES.items():
        df = load_and_preprocess_data(file_path)
        calibration_df, val_df = split_data(df)
        threshold = multistart_fixed_sequence_threshold_selection(calibration_df, TARGET_cFAR)

        print(f"\n=== {scheme_name} (Multistart FS) ===")
        results = []
        for r in RISK_TARGETS:
            threshold_r = multistart_fixed_sequence_threshold_selection(calibration_df, r)
            if threshold_r is None:
                cal_accept = val_accept = 0.0
                val_cfar = np.nan
                val_pm = cfar_pm = np.nan
            else:
                cal_accept = (calibration_df["uncertainty"] <= threshold_r).mean()
                val_stats = validate_with_ci(val_df, threshold_r)
                val_accept = val_stats["val_accept_rate"]
                val_cfar = val_stats["validation_cfar"]
                val_pm = ci_to_pm(val_stats["val_accept_ci"])
                cfar_pm = ci_to_pm(val_stats["validation_cfar_ci"])
            results.append({
                "risk_target": r,
                "cal_threshold": threshold_r if threshold_r is not None else "reject-all",
                "cal_accept_rate": cal_accept,
                "val_accept_rate": val_accept,
                "val_accept_pm": val_pm,
                "val_cfar": val_cfar,
                "val_cfar_pm": cfar_pm
            })
        print(pd.DataFrame(results).to_string(index=False))

        # plot style
        if scheme_name.startswith("Scheme A"):
            style = {"color": "#1f77b4", "marker": "s", "linestyle": "-"}
        elif scheme_name.startswith("Scheme B w/ norms"):
            style = {"color": "#2ca02c", "marker": "^", "linestyle": "-"}
        else:
            style = {"color": "#ff7f0e", "marker": "o", "linestyle": "-"}
        style_map[scheme_name] = style

        # ----- calibration CFAR plot -----
        th, cfar_vals, _, _ = compute_cfar_curve_with_ci(calibration_df)
        plt.plot(
            th, cfar_vals, linewidth=2, label=scheme_name,
            color=style["color"], linestyle=style["linestyle"],
            marker=style["marker"], markevery=15
        )

        # threshold line & label
        if threshold is not None:
            idx = np.abs(th - threshold).argmin()
            cfar_at_tau = cfar_vals[idx]
            plt.vlines(threshold, 0, cfar_at_tau, linestyles="--", linewidth=1.8, color=style["color"])
            
            if "Scheme A" in scheme_name:
                x_text, y_text, ha, va = threshold, -0.005, "center", "top"
            elif "Scheme B w/ norms" in scheme_name:
                x_text, y_text, ha, va = threshold + 0.009, 0.01, "left", "bottom"
            else:
                x_text, y_text, ha, va = threshold + 0.02, 0.05, "left", "bottom"

            plt.text(x_text, y_text, f"{threshold:.2f}", color=style["color"], fontsize=10, ha=ha, va=va)

            # validation marker
            val_th, val_cfars = construct_empirical_far_curve_validation(val_df)
            idx_val = np.abs(val_th - threshold).argmin()
            val_points[scheme_name] = (threshold, val_cfars[idx_val])

    # ----- finalize plot -----
    plt.axhline(y=TARGET_cFAR, color="red", linestyle="--", linewidth=2)
    plt.text(1.0, TARGET_cFAR + 0.005, f"target CFAR = {TARGET_cFAR:.2f}", color="red",
                 ha="right", va="bottom", fontsize=11)
    plt.xlabel("Model Uncertainty", fontweight="bold")
    plt.ylabel("Cumulative False Answer Rate (CFAR)", fontweight="bold")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.legend()

    # validation markers
    for scheme_name, (threshold, val_cfar) in val_points.items():
        color = style_map[scheme_name]["color"]
        plt.scatter(threshold, val_cfar, marker="x", s=70, linewidths=2, color=color, zorder=5)
        plt.vlines(threshold, ymin=min(val_cfar, TARGET_cFAR),
                   ymax=max(val_cfar, TARGET_cFAR),
                   colors=color, linestyles=":", linewidth=1.8, alpha=0.9)

    plt.savefig(os.path.join(RESULT_DIR, f"mfs_cfar_plot_calibration_ratio_{CALIBRATION_RATIO}_{TIMESTAMP}.png"),
                dpi=200, bbox_inches="tight")
    plt.close()
    print("Plots saved.")


if __name__ == "__main__":
    FILES = load_files(model_name, correct, incorrect, abstain)
    for L in START_L_LIST:
        START_POINTS = np.linspace(0, len(THRESHOLDS)-1, L, dtype=int)

        print(f"\n===== L = {L} starts =====")
        run_all_schemes()
