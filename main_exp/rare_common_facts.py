import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from scipy.stats import gaussian_kde, binom
import glob
from typing import Tuple, Dict

model_name = "gpt-5-mini"
correct = "+1"
incorrect = "-1"
abstain = "+0.4"

RESULT_DIR = f"outputs/{model_name}/rare_common_facts"
os.makedirs(RESULT_DIR, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
Z_95 = 1.96


def load_files(model_name, correct, incorrect, abstain):
    base_dir = f"outputs/{model_name}"
    scheme_info = {
        f"Scheme A ({correct}, {incorrect})": f"popqa_A_{correct}_{incorrect}_*.csv",
        f"Scheme B ({correct}, {incorrect}, {abstain})": f"popqa_B_{correct}_{incorrect}_{abstain}_*.csv",
        f"Scheme B Norms ({correct}, {incorrect}, {abstain})": f"popqa_B_norm_{correct}_{incorrect}_{abstain}_*.csv"
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

plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 22,
    'axes.labelsize': 16,
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14
})


def split_by_o_pop(df: pd.DataFrame):
    """
    Sort by o_pop and split into upper 1/3 (common) and lower 1/3 (rare)
    """
    df = df.copy()
    df["o_pop"] = pd.to_numeric(df["o_pop"], errors="coerce")
    df = df.dropna(subset=["o_pop"])

    df = df.sort_values("o_pop", ascending=False).reset_index(drop=True)

    n = len(df)
    one_third = n // 3

    df_common = df.iloc[:one_third]
    df_rare = df.iloc[-one_third:]

    return df_common, df_rare

def plot_confidence_by_correctness(df, file_label, pop_label):
    """
    Plots two side-by-side histograms showing confidence distribution separated by correctness (correct=0 vs correct=1).
    For Experiment B, combines first_confidence and best_guess_confidence.
    """
    df_plot = df.copy()
    df_plot.loc[df_plot['idk_flag'] == 0, 'confidence'] = df_plot.loc[df_plot['idk_flag'] == 0, 'first_confidence']
    df_plot.loc[df_plot['idk_flag'] == 1, 'confidence'] = df_plot.loc[df_plot['idk_flag'] == 1, 'best_guess_confidence']
    
    # Clean data
    df_plot['confidence'] = pd.to_numeric(df_plot['confidence'], errors='coerce')
    df_plot = df_plot.dropna(subset=['confidence'])
    df_plot['confidence'] = df_plot['confidence'].clip(0, 1)
    
    # Create bins for 0-1 range (20 bins of 0.05 each)
    bins = np.linspace(0, 1, 21)
    bin_edges = list(bins)
    
    # Split by correctness
    correct_data = df_plot.loc[df_plot['correct'] == 1, 'confidence']
    incorrect_data = df_plot.loc[df_plot['correct'] == 0, 'confidence']

    # Create single plot with both distributions
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    x_smooth = np.linspace(0, 1, 300)
    max_density = 0

    # Compute proportions so that correct + incorrect integrate to 1
    n_correct = len(correct_data)
    n_incorrect = len(incorrect_data)
    n_total = n_correct + n_incorrect
    p_correct = n_correct / n_total if n_total > 0 else 0
    p_incorrect = n_incorrect / n_total if n_total > 0 else 0

    # Plot correct answers (green) scaled by p_correct
    density_correct = np.zeros_like(x_smooth)
    if n_correct > 0:
        kde_correct = gaussian_kde(correct_data.values, bw_method='scott')
        density_correct = kde_correct(x_smooth) * p_correct
        max_density = max(max_density, density_correct.max())
        ax.plot(x_smooth, density_correct, color='green', linewidth=2,
                label=f'Correct')
        ax.fill_between(x_smooth, density_correct, alpha=0.3, color='green')

    # Plot incorrect answers (red) scaled by p_incorrect
    density_incorrect = np.zeros_like(x_smooth)
    if n_incorrect > 0:
        kde_incorrect = gaussian_kde(incorrect_data.values, bw_method='scott')
        density_incorrect = kde_incorrect(x_smooth) * p_incorrect
        max_density = max(max_density, density_incorrect.max())
        ax.plot(x_smooth, density_incorrect, color='red', linewidth=2,
                label=f'Incorrect')
        ax.fill_between(x_smooth, density_incorrect, alpha=0.3, color='red')      

    ax.set_title(f"Confidence Distribution with Correctness\n{file_label} – {pop_label}", fontsize=16, fontweight='bold')
    ax.set_xlabel("Confidence (0-1)")
    ax.set_ylabel("Density")
    ax.set_ylim(0, 4)
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0,1,11))
    ax.grid(axis='y',linestyle='--',alpha=0.7)
    ax.legend(loc='upper right')

    plt.tight_layout()

    output_dir = f"outputs/{model_name}/rare_common_facts/idk_confidence_density"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"idk_confidence_hist_{file_label}_{pop_label}_{TIMESTAMP}.png"
    plt.savefig(os.path.join(output_dir,filename),dpi=300)
    plt.close()
    print(f"✅ Saved confidence by correctness plot for: {filename}")

def plot_idk_confidence_separate(df, exp_label, pop_label):
    """
    Plots separate histograms for Answered (IDK=0) and Said IDK/Best Guess (IDK=1).
    Each plot shows correct vs incorrect breakdown as stacked bars.
    """
    
    df_plot = df.copy()
    df_plot.loc[df_plot['idk_flag'] == 0, 'confidence'] = df_plot.loc[df_plot['idk_flag'] == 0, 'first_confidence']
    df_plot.loc[df_plot['idk_flag'] == 1, 'confidence'] = df_plot.loc[df_plot['idk_flag'] == 1, 'best_guess_confidence']
    idk1_label = "Best Guess"

    # Clean data
    df_plot['confidence'] = pd.to_numeric(df_plot['confidence'], errors='coerce')
    df_plot = df_plot.dropna(subset=['confidence'])
    df_plot['confidence'] = df_plot['confidence'].clip(0, 1)

    # Create bins for 0-1 range (20 bins of 0.05 each)
    bins = np.linspace(0, 1, 21)
    bin_edges = list(bins)
    
    # Get total count for normalization
    total_count = len(df_plot)
    
    # Create two separate plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ---- Plot 1: Answered (IDK=0) ----
    idk_0 = df_plot[df_plot['idk_flag'] == 0]
    if len(idk_0) > 0:
        idk_0_correct = idk_0.loc[idk_0['correct'] == 1, 'confidence']
        idk_0_incorrect = idk_0.loc[idk_0['correct'] == 0, 'confidence']
        
        hist_0_correct, _ = np.histogram(idk_0_correct, bins=bin_edges)
        hist_0_incorrect, _ = np.histogram(idk_0_incorrect, bins=bin_edges)
        
        # Normalize by total dataset count (not just answered)
        hist_0_correct_norm = hist_0_correct / total_count
        hist_0_incorrect_norm = hist_0_incorrect / total_count
        
        bin_centers = [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]
        width = 0.04  # bar width (adjusted for 0-1 scale)
        
        ax1.bar(bin_centers, hist_0_correct_norm, width=width,
               label='Correct', color='green', alpha=0.7, edgecolor='black')
        ax1.bar(bin_centers, hist_0_incorrect_norm, width=width, bottom=hist_0_correct_norm,
               label='Incorrect', color='red', alpha=0.7, edgecolor='black')
  
        ax1.set_title(f'Answered ({exp_label} - {pop_label})', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Confidence (0-1)')
        ax1.set_ylabel('Proportion of Total Dataset')
        ax1.set_ylim(0, 0.6)
        ax1.legend()
        ax1.set_xticks(np.linspace(0, 1, 11))  # 0, 0.1, 0.2, ..., 1.0
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
    else:
        ax1.text(0.5, 0.5, 'No data for Answered', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title(f'Answered ({exp_label} - {pop_label})', fontsize=16, fontweight='bold')
    
    # ---- Plot 2: Said IDK / Best Guess (IDK=1) ----
    idk_1 = df_plot[df_plot['idk_flag'] == 1]
    if len(idk_1) > 0:
        idk_1_correct = idk_1.loc[idk_1['correct'] == 1, 'confidence']
        idk_1_incorrect = idk_1.loc[idk_1['correct'] == 0, 'confidence']
        
        hist_1_correct, _ = np.histogram(idk_1_correct, bins=bin_edges)
        hist_1_incorrect, _ = np.histogram(idk_1_incorrect, bins=bin_edges)
        
        # Normalize by total dataset count (not just IDK)
        hist_1_correct_norm = hist_1_correct / total_count
        hist_1_incorrect_norm = hist_1_incorrect / total_count
        
        bin_centers = [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]
        width = 0.04  # bar width (adjusted for 0-1 scale)
        
        ax2.bar(bin_centers, hist_1_correct_norm, width=width,
                   label='Correct', color='lightgreen', alpha=0.7, edgecolor='black')
        ax2.bar(bin_centers, hist_1_incorrect_norm, width=width, bottom=hist_1_correct_norm,
               label='Incorrect', color='lightcoral', alpha=0.7, edgecolor='black')
            
        
        ax2.set_title(f'{idk1_label} ({exp_label} - {pop_label})', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Confidence (0-1)')
        ax2.set_ylabel('Proportion of Total Dataset')
        ax2.set_ylim(0, 0.6)
        ax2.legend()
        ax2.set_xticks(np.linspace(0, 1, 11))  # 0, 0.1, 0.2, ..., 1.0
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
    else:
        ax2.text(0.5, 0.5, f'No data for {idk1_label}', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title(f'{idk1_label} ({exp_label} - {pop_label})', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    output_dir = f"outputs/{model_name}/rare_common_facts/plot_idk_confidence_separate"
    os.makedirs(output_dir, exist_ok=True)
    
    output_name = f"idk_confidence_separate_{exp_label}_{pop_label}_{TIMESTAMP}.png"
    output_path = os.path.join(output_dir, output_name)
    
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"✅ Saved separate confidence histograms for file {exp_label}_{pop_label} → {output_path}")


def calculate_ci(successes, trials, confidence=0.95):
    if trials == 0:
        return (0, 0)
    lower = binom.ppf((1 - confidence) / 2, trials, successes / trials) / trials
    upper = binom.ppf(1 - (1 - confidence) / 2, trials, successes / trials) / trials
    return (lower, upper)


# Modify confidence interval output to show +/- format
def calculate_ci_pm(successes, trials, confidence=0.95):
    if trials == 0:
        return 0
    p_hat = successes / trials
    margin = binom.ppf(1 - (1 - confidence) / 2, trials, p_hat) / trials - p_hat
    return margin

def summarize(df, label, file_handle=None):

    acc = df["correct"].mean()
    answered = df["idk_flag"] == 0 
    far_e1 = df.loc[answered, "false_answer_flag"].sum() / answered.sum()
    far_e2 = 1 - acc

    far_e1_margin = calculate_ci_pm(df.loc[answered, "false_answer_flag"].sum(), answered.sum())
    far_e2_margin = calculate_ci_pm(len(df) - df["correct"].sum(), len(df))

    summary_text = (
        f"\n📊 {label} Summary:\n"
        f"  Number of Answered Questions: {answered.sum()}\n"
        f"  Number of Incorrect Answers: {df.loc[answered, 'false_answer_flag'].sum()}\n"
        f"  False-Answer Rate - E1: {far_e1:.3f} ± {far_e1_margin:.3f}\n"
        f"  False-Answer Rate - E2: {far_e2:.3f} ± {far_e2_margin:.3f}\n"
    )
    
    if file_handle is not None:
        file_handle.write(summary_text)


def calculate_total_reward(df, label=""):
    """
    Calculate total reward by summing the 'score' column.
    """
    if "score" not in df.columns:
        raise ValueError("❌ Column 'score' not found in dataframe")

    total_reward = df["score"].sum()

    return total_reward

def calculate_aer(df, label=""):
    """
    Abstention-to-Error Ratio (AER) = #abstain / #incorrect
    """
    n_abstain = (df["idk_flag"] == 1).sum()
    n_incorrect = (df["correct"] == 0).sum()

    if n_incorrect == 0:
        aer = float("inf")  # or set to 0 if you prefer
    else:
        aer = n_abstain / n_incorrect

    return aer

def calculate_brier_score(df: pd.DataFrame) -> float:
    """
    Brier score = mean((confidence - correctness)^2)
    """
    df['confidence'] = df.apply(
        lambda row: row['first_confidence'] if row['idk_flag'] == 0 else row['best_guess_confidence'], axis=1
    )
    df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')
    df['confidence'] = df['confidence'].clip(0, 1)

    brier_score = np.mean((df['confidence'] - df['correct']) ** 2)
    return brier_score

def calculate_brier_score_with_ci(df: pd.DataFrame, confidence: float = 0.95, answered_only: bool = False) -> Tuple[float, float]:
    """
    Calculate the Brier score and its confidence interval for the dataset.

    Args:
        df: DataFrame containing the data.
        confidence: Confidence level for the interval (default: 95%).

    Returns:
        A tuple containing the Brier score and the margin of error (Brier score ± margin).
    """
    # If requested, restrict to answered rows only and use their first_confidence.
    if answered_only:
        df = df[df['idk_flag'] == 0].copy()
        df['confidence'] = pd.to_numeric(df['first_confidence'], errors='coerce')
    else:
        df['confidence'] = df.apply(
            lambda row: row['first_confidence'] if row['idk_flag'] == 0 else row['best_guess_confidence'], axis=1
        )
    df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')
    df['confidence'] = df['confidence'].clip(0, 1)

    errors = (df['confidence'] - df['correct']) ** 2
    errors = errors.dropna()  # Remove NaN values

    if len(errors) == 0:
        return float('nan'), float('nan')

    brier_score = np.mean(errors)

    # Calculate confidence interval using binomial approximation
    def calculate_ci_pm(successes, trials, confidence=0.95):
        if trials == 0:
            return 0
        p_hat = successes / trials
        margin = binom.ppf(1 - (1 - confidence) / 2, trials, p_hat) / trials - p_hat
        return margin

    n = len(errors)
    # For the margin calculation we treat the sum of squared-errors as "successes"
    # in a binomial approximation (legacy choice). If you want a different
    # CI (e.g. bootstrap), we can change this.
    margin = calculate_ci_pm(np.sum(errors), n, confidence)

    return brier_score, margin

def calculate_calibration_error(df: pd.DataFrame, file_label: str, num_bins: int = 10) -> Dict[str, float]:
    """
    Calculate Expected Calibration Error (ECE) = (1/m) * sum over bins: m_b * |o_hat_b - p_hat_b|
    
    where:
    - m is total number of samples
    - m_b is number of samples in bin b
    - o_hat_b is empirical frequency (accuracy) in bin b
    - p_hat_b is average predicted probability (self reported confidence) in bin b
    
    For file_label 'B', uses best_guess_confidence for IDK=1; otherwise uses first_confidence for both.
    """
    
    df_calc = df.copy()
    df_calc.loc[df_calc['idk_flag'] == 0, 'confidence'] = df_calc.loc[df_calc['idk_flag'] == 0, 'first_confidence']
    df_calc.loc[df_calc['idk_flag'] == 1, 'confidence'] = df_calc.loc[df_calc['idk_flag'] == 1, 'best_guess_confidence']
    
    # Clean data
    df_calc['confidence'] = df_calc['confidence'] 
    df_calc['confidence'] = df_calc['confidence'].clip(0, 1)

    # Create bins
    bins = np.linspace(0, 1, num_bins + 1)
    df_calc['bin'] = pd.cut(df_calc['confidence'], bins=bins, include_lowest=True, labels=False)
    
    # Calculate per-bin statistics
    bin_stats = []
    ece = 0.0
    total_samples = len(df_calc)  # m in the formula
    
    for bin_idx in range(num_bins):
        bin_data = df_calc[df_calc['bin'] == bin_idx]
        
        if len(bin_data) > 0:
            bin_count = len(bin_data)  # m_b in the formula
            
            avg_confidence = bin_data['confidence'].mean()

            empirical_frequency = bin_data['correct'].mean()
            
            calibration_gap = abs(empirical_frequency - avg_confidence)
            
            ece += (bin_count / total_samples) * calibration_gap
            
            bin_stats.append({
                'bin_idx': bin_idx,
                'bin_range': f'[{bins[bin_idx]:.2f}, {bins[bin_idx+1]:.2f}]',
                'count': bin_count,
                'avg_predicted_prob': avg_confidence,  # p_hat_b
                'empirical_frequency': empirical_frequency,  # o_hat_b
                'calibration_gap': calibration_gap
            })
        else:
            bin_stats.append({
                'bin_idx': bin_idx,
                'bin_range': f'[{bins[bin_idx]:.2f}, {bins[bin_idx+1]:.2f}]',
                'count': 0,
                'avg_predicted_prob': np.nan,
                'empirical_frequency': np.nan,
                'calibration_gap': np.nan
            })
    
    return {
        'ECE': ece,
        'total_samples': total_samples,
        'num_bins': num_bins,
        'bin_stats': bin_stats
    }


def calculate_calibration_by_category(df: pd.DataFrame, file_label: str, num_bins: int = 10) -> Dict[str, Dict]:
    """
    Calculate calibration error separately for answered questions (idk_flag=0) and all questions combined.
    
    Returns:
        Dictionary with separate calibration metrics for each category
    """
    
    # Split by IDK flag
    df_answered = df[df['idk_flag'] == 0].copy()
    
    results = {}
    
    # Calculate for answered questions
    if len(df_answered) > 0:
        results['answered'] = calculate_calibration_error(df_answered, file_label, num_bins)
    else:
        results['answered'] = None
    
    # Calculate for all combined
    results['overall'] = calculate_calibration_error(df, file_label, num_bins)
    
    return results


def print_calibration_report(results: Dict[str, Dict], label: str, file_handle=None):

    for category, metrics in results.items():
        if metrics is None:
            continue
        
        text = (
            f"{label} - {category.upper()} ECE: {metrics['ECE']:.4f}\n"
        )

        if file_handle:
            file_handle.write(text)


def save_calibration_report_csv(results: Dict[str, Dict], label: str, output_path: str):
    """    
    Args:
        results: Dictionary returned by calculate_calibration_by_category
        label: Label for the experiment (e.g., 'Experiment A')
        output_path: Path to save the CSV file
    """
    all_rows = []
    
    for category, metrics in results.items():
        if metrics is None:
            continue
        
        # Add summary row
        all_rows.append({
            'experiment': label,
            'category': category.upper(),
            'metric': 'ECE',
            'value': metrics['ECE'],
            'total_samples': metrics['total_samples'],
            'num_bins': metrics['num_bins'],
            'bin_range': '',
            'bin_count': '',
            'avg_predicted_prob': '',
            'empirical_frequency': '',
            'calibration_gap': ''
        })
        
        # Add per-bin rows
        for bin_stat in metrics['bin_stats']:
            if bin_stat['count'] > 0:
                all_rows.append({
                    'experiment': label,
                    'category': category.upper(),
                    'metric': 'bin_detail',
                    'value': '',
                    'total_samples': '',
                    'num_bins': '',
                    'bin_range': bin_stat['bin_range'],
                    'bin_count': bin_stat['count'],
                    'avg_predicted_prob': bin_stat['avg_predicted_prob'],
                    'empirical_frequency': bin_stat['empirical_frequency'],
                    'calibration_gap': bin_stat['calibration_gap']
                })
    
    df_report = pd.DataFrame(all_rows)
    df_report.to_csv(output_path, index=False)
    print(f"\n💾 Calibration report saved to: {output_path}")

def main():
    summary_path = os.path.join(RESULT_DIR, "summary.txt")

    # --- Per-scheme loop for summaries and per-scheme plots ---
    with open(summary_path, "w", encoding="utf-8") as f:
        for scheme_key, file_list in FILES.items():
            scheme_label = scheme_key.replace("_", " ")  # Nice label for plots and summaries
            f.write(f"\n===== {scheme_label} =====\n")

            df_all = pd.read_csv(file_list)
            df_common, df_rare = split_by_o_pop(df_all)

            # Add category column for later combined plot
            df_common["category"] = "common"
            df_rare["category"] = "rare"

            # Summaries
            summarize(df_common, f"{scheme_label} - Common Facts", file_handle=f)
            summarize(df_rare, f"{scheme_label} - Rare Facts", file_handle=f)

            # --- Total Reward ---
            reward_common = calculate_total_reward(df_common, f"{scheme_label} - Common Facts")
            reward_rare = calculate_total_reward(df_rare, f"{scheme_label} - Rare Facts")

            f.write(f"\n🏆 {scheme_label} - Common Total Reward: {reward_common:.2f}\n")
            f.write(f"🏆 {scheme_label} - Rare Total Reward: {reward_rare:.2f}\n")

            # --- Abstention-to-Error Ratio (AER) ---
            aer_common = calculate_aer(df_common, f"{scheme_label} - Common Facts")
            aer_rare = calculate_aer(df_rare, f"{scheme_label} - Rare Facts")

            f.write(f"\n📉 {scheme_label} - Common AER: {aer_common:.3f}\n")
            f.write(f"📉 {scheme_label} - Rare AER: {aer_rare:.3f}\n")

            # Overall Brier scores (use first_confidence or best_guess for IDK rows)
            brier_common, margin_common = calculate_brier_score_with_ci(df_common)
            brier_rare, margin_rare = calculate_brier_score_with_ci(df_rare)

            # Brier scores restricted to answered rows only (idk_flag == 0)
            brier_score_common_answered, margin_common_answered = calculate_brier_score_with_ci(df_common, answered_only=True)
            brier_score_rare_answered, margin_rare_answered = calculate_brier_score_with_ci(df_rare, answered_only=True)

            f.write(f"\n{scheme_label} - Common facts:\n")
            f.write(f"\nBrier Score (overall) = {brier_common:.4f} ± {margin_common:.4f}\n")
            f.write(f"Brier Score (answered only) = {brier_score_common_answered:.4f} ± {margin_common_answered:.4f}\n")

            f.write(f"\n{scheme_label} - Rare facts:\n")
            f.write(f"\nBrier Score (overall) = {brier_rare:.4f} ± {margin_rare:.4f}\n")
            f.write(f"Brier Score (answered only) = {brier_score_rare_answered:.4f} ± {margin_rare_answered:.4f}\n\n")

            # ECE            
            # Common facts
            results_common = calculate_calibration_by_category(df_common, scheme_label, num_bins=10)
            print_calibration_report(results_common, f"{scheme_label} – COMMON", file_handle=f)

            ece_dir = os.path.join(RESULT_DIR, "ece_reports")
            os.makedirs(ece_dir, exist_ok=True)

            save_calibration_report_csv(
                results_common,
                f"{scheme_label}_COMMON",
                os.path.join(ece_dir, f"ece_{scheme_label}_common_{TIMESTAMP}.csv")
            )

            # Rare facts
            results_rare = calculate_calibration_by_category(df_rare, scheme_label, num_bins=10)
            print_calibration_report(results_rare, f"{scheme_label} – RARE", file_handle=f)

            save_calibration_report_csv(
                results_rare,
                f"{scheme_label}_RARE",
                os.path.join(ece_dir, f"ece_{scheme_label}_rare_{TIMESTAMP}.csv")
            )

            # Per-scheme confidence plots
            plot_confidence_by_correctness(df_common, scheme_label, "Common")
            plot_confidence_by_correctness(df_rare, scheme_label, "Rare")

            plot_idk_confidence_separate(df_common, scheme_label, "Common")
            plot_idk_confidence_separate(df_rare, scheme_label, "Rare")

    # --- Cross-scheme combined confidence comparison (once) ---
    df_A = pd.read_csv(FILES[f"Scheme A ({correct}, {incorrect})"])
    df_B = pd.read_csv(FILES[f"Scheme B ({correct}, {incorrect}, {abstain})"])
    df_B_norm = pd.read_csv(FILES[f"Scheme B Norms ({correct}, {incorrect}, {abstain})"])

    # Split into common/rare and add category column
    for df in [df_A, df_B, df_B_norm]:
        df_common, df_rare = split_by_o_pop(df)
        df_common["category"] = "common"
        df_rare["category"] = "rare"
        df_combined = pd.concat([df_common, df_rare], ignore_index=True)

        # Update original df variable
        if df is df_A:
            df_A = df_combined
        elif df is df_B:
            df_B = df_combined
        else:
            df_B_norm = df_combined

    print(f"\n✅ Summary saved to {summary_path}")

if __name__ == "__main__":

    FILES = load_files(
        model_name,
        correct,
        incorrect,
        abstain
    )

    main()
