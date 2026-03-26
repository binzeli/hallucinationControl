import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from datetime import datetime
from math import sqrt
import glob
from scipy.stats import norm, beta

TARGET_cFAR = 0.3
THRESHOLDS = np.arange(0, 1.01, 0.01)  # {0,0.01,...,1.00}
NUM_THRESHOLDS = len(THRESHOLDS)

CALIBRATION_RATIO = 0.2 # 1 for full dataset
RANDOM_SEED = 1

model_name = "gpt-5-mini"
correct = "+1"
incorrect = "-1"
abstain = "+0.4"

RESULT_DIR = f"main_exp/outputs/{model_name}/cfar_plots"
os.makedirs(RESULT_DIR, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

DELTA = 0.05 # 95% confidence
ALPHA = DELTA / NUM_THRESHOLDS # Bonferroni correction

# Set multiple risk targets
RISK_TARGETS = [0.1, 0.2, 0.3, 0.4]

def load_files(model_name, correct, incorrect, abstain):
    base_dir = f"main_exp/example_output/{model_name}"
    scheme_info = {
        f"Scheme A ({correct}, {incorrect})": f"popqa_A_{correct}_{incorrect}_*.csv",
        f"Scheme B ({correct}, {incorrect}, {abstain})": f"popqa_B_{correct}_{incorrect}_{abstain}_*.csv",
        f"Scheme B w/ norms ({correct}, {incorrect}, {abstain})": f"popqa_B_norm_{correct}_{incorrect}_{abstain}_*.csv"
    }
    FILES = {}
    for scheme_name, pattern in scheme_info.items():
        latest_file = get_latest_file(base_dir, pattern)
        FILES[scheme_name] = latest_file
    return FILES


def get_latest_file(folder, pattern):
    files = glob.glob(os.path.join(folder, pattern))
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    return max(files, key=os.path.getmtime)


def wilson_ci(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return center - margin, center + margin

def clopper_pearson_upper(k, n, alpha):
    if n == 0 or k == n:
        return 1.0
    return beta.ppf(1 - alpha, k + 1, n - k)

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

def select_ucb_threshold(curve_df, target_cfar):
    valid = curve_df
    valid = valid[valid["ucb"] <= target_cfar]

    if len(valid) == 0:
        print(f"[Warning] No certified threshold for target CFAR={target_cfar}")
        return None

    return valid["threshold"].max()

def construct_empirical_far_curve_calibration(df):
    thresholds = THRESHOLDS.copy()
    records = []

    for t in thresholds:
        accepted = df[df["uncertainty"] <= t]

        n_acc = len(accepted)
        k_false = int((1 - accepted["correct"]).sum())

        cfar = compute_cfar(accepted)
        
        if n_acc == 0:
            ucb = 1.0
        else:
            ucb = clopper_pearson_upper(k_false, n_acc, ALPHA)

        records.append({
            "threshold": t,
            "cfar": cfar,
            "num_accepted": n_acc,
            "num_false": k_false,
            "ucb": ucb
        })

    return pd.DataFrame(records)

def construct_empirical_far_curve_validation(df):
    thresholds = THRESHOLDS.copy()
    cfars = [compute_cfar(df[df["uncertainty"] <= t]) for t in thresholds]
    return thresholds, np.array(cfars)


def select_empirical_threshold(curve_df, target_cfar):
    valid = curve_df[curve_df["cfar"] <= target_cfar]
    if len(valid) == 0: raise ValueError("No threshold satisfies target FAR.")
    return valid["threshold"].max()


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


# Compute actual empirical CFAR curve (no bootstrap)
def compute_true_curve(df):
    thresholds = THRESHOLDS.copy()
    curve = [compute_cfar(df[df["uncertainty"] <= t]) for t in thresholds]
    return thresholds, np.array(curve)

def compute_cfar_curve_with_ci(df):
    thresholds = THRESHOLDS.copy()

    cfar_vals = []
    ci_low = []
    ci_high = []

    for t in thresholds:
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

    cfar_vals = np.array(cfar_vals)
    ci_low = np.array(ci_low)
    ci_high = np.array(ci_high)

    # force last point = whole dataset
    true_cfar = compute_cfar(df)
    cfar_vals[-1] = true_cfar
    thresholds[-1] = 1.0

    return thresholds, cfar_vals, ci_low, ci_high

def run_all_schemes():
    fig_calibration = plt.figure(figsize=(9,6))
    val_points = {}  # list to store validation markers
    style_map = {}

    if CALIBRATION_RATIO < 1.0:
        fig_val = plt.figure(figsize=(9,6))

    for scheme_name, file_path in FILES.items():
        df = load_and_preprocess_data(file_path)
        calibration_df, val_df = split_data(df)

        curve_df = construct_empirical_far_curve_calibration(calibration_df)
        threshold = select_ucb_threshold(curve_df, TARGET_cFAR)
        
        if CALIBRATION_RATIO < 1.0:
            # multiple risk target
            results = []

            for r in RISK_TARGETS:

                threshold_r = select_ucb_threshold(curve_df, r)

                if threshold_r is None:
                    calibration_coverage = 0.0
                    val_coverage = 0.0
                    cfar = np.nan
                    val_coverage_pm = 0.0
                    cfar_pm = np.nan
                else:
                    calibration_coverage = (calibration_df["uncertainty"] <= threshold_r).mean()

                    val_stats = validate_with_ci(val_df, threshold_r)

                    val_coverage = val_stats["val_accept_rate"]
                    cfar = val_stats["validation_cfar"]

                    val_coverage_pm = ci_to_pm(val_stats["val_accept_ci"])
                    cfar_pm = ci_to_pm(val_stats["validation_cfar_ci"])

                results.append({
                    "risk_target": r,
                    "cal_threshold": threshold_r if threshold_r is not None else "reject-all",
                    "cal_accept_rate": calibration_coverage, 
                    "val_accept_rate": val_coverage,
                    "val_accept_pm": val_coverage_pm,
                    "val_cfar": cfar,
                    "val_cfar_pm": cfar_pm
                })

            results = pd.DataFrame(results)
            print(f"\nRisk-Coverage Table ({scheme_name})")
            print(results.to_string(index=False))
        
        if scheme_name.startswith("Scheme A"):
            style = {"color": "#1f77b4", "marker": "s", "linestyle": "-"}
        elif scheme_name.startswith("Scheme B w/ norms"):
            style = {"color": "#2ca02c", "marker": "^", "linestyle": "-"}
        else:
            style = {"color": "#ff7f0e", "marker": "o", "linestyle": "-"}
        style_map[scheme_name] = style

        # ----- calibration plot -----
        th, cfar_vals, ci_low, ci_high = compute_cfar_curve_with_ci(calibration_df)
        plt.figure(fig_calibration.number)
        plt.plot(
            th,
            cfar_vals,
            linewidth=2,
            label=scheme_name,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markevery=15
        )

        if CALIBRATION_RATIO < 1.0:
            valid = curve_df
            plt.plot(
                valid["threshold"],
                valid["ucb"],
                linestyle="--",
                linewidth=1,
                color=style["color"],
                alpha=0.8,
                label=f"{scheme_name} CP-UCB"
            )
        if CALIBRATION_RATIO == 1.0:
            plt.fill_between(
                th,
                ci_low,
                ci_high,
                alpha=0.1,
                color=style["color"]
            )

        if CALIBRATION_RATIO < 1.0:
            # ----- Empirical τ line and label -----
            if threshold is not None:
                idx = np.abs(curve_df["threshold"] - threshold).argmin()
                cfar_at_tau = curve_df.iloc[idx]["cfar"]
                plt.vlines(threshold, 0, cfar_at_tau, linestyles="--", linewidth=1.8, color=style["color"])

            if "Scheme A" in scheme_name:
                x_text, y_text, ha, va = threshold, -0.005, "center", "top"
            elif "Scheme B w/ norms" in scheme_name:
                x_text, y_text, ha, va = threshold + 0.009, 0.01, "left", "bottom"
            else:
                x_text, y_text, ha, va = threshold + 0.02, 0.05, "left", "bottom"

            if threshold is not None:
                plt.text(x_text, y_text, f"{threshold:.2f}", color=style["color"], fontsize=10, ha=ha, va=va)

        print(f"\n=== {scheme_name} ===")
        if threshold is None:
            print("selected_threshold: REJECT-ALL")
        else:
            print(f"selected_threshold: {threshold:.4f}")

        if CALIBRATION_RATIO == 1.0: 
            true_cfar = compute_cfar(calibration_df)
            print(f"CFAR for whole dataset: {true_cfar:.6f}")

        # ----- Validation info -----
        if CALIBRATION_RATIO < 1.0:
            val_th, val_cfars = construct_empirical_far_curve_validation(val_df)
            if threshold is not None:
                idx = np.abs(val_th - threshold).argmin()
                cfar_at_tau_val = val_cfars[idx]
                val_points[scheme_name] = (threshold, cfar_at_tau_val)

            if threshold is None:
                print("reject-all → no validation stats")
                calibration_ar = 0.0
            else:
                val_stats = validate_with_ci(val_df, threshold)

                calibration_ar = (calibration_df["uncertainty"] <= threshold).mean()
                val_ar_pm = ci_to_pm(val_stats["val_accept_ci"])
                val_cfar_pm = ci_to_pm(val_stats["validation_cfar_ci"])

                print(f"calibration_accept_rate   : {calibration_ar:.4f}")
                print(f"val_accept_rate     : {val_stats['val_accept_rate']:.4f} ± {val_ar_pm:.4f}")
                print(f"validation_cfar     : {val_stats['validation_cfar']:.4f} ± {val_cfar_pm:.4f}")

    # ----- Finalize calibration plot -----
    plt.figure(fig_calibration.number)
    
    if CALIBRATION_RATIO < 1.0:
        plt.axhline(y=TARGET_cFAR, color="red", linestyle="--", linewidth=2)
        plt.text(1.0, TARGET_cFAR + 0.005, f"target CFAR = {TARGET_cFAR:.2f}", color="red",
                 ha="right", va="bottom", fontsize=11)

    plt.xlabel("Model Uncertainty", fontweight="bold")
    plt.ylabel("Cumulative False Answer Rate (CFAR)", fontweight="bold")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()

    # Validation markers on calibration plot
    if CALIBRATION_RATIO < 1.0:
        for scheme_name, (threshold, val_cfar) in val_points.items():
            color = style_map[scheme_name]["color"]
            plt.scatter(threshold, val_cfar, marker="x", s=70, linewidths=2, color=color, zorder=5)
            plt.vlines(threshold, ymin=min(val_cfar, TARGET_cFAR), ymax=max(val_cfar, TARGET_cFAR),
                       colors=color, linestyles=":", linewidth=1.8, alpha=0.9)

    plt.savefig(os.path.join(RESULT_DIR, f"cfar_plot_calibration_ratio_{CALIBRATION_RATIO}_{TIMESTAMP}.png"),
                dpi=200, bbox_inches="tight")
    plt.close()
    print("Plots saved.")


if __name__ == "__main__":
    FILES = load_files(model_name, correct, incorrect, abstain)
    run_all_schemes()