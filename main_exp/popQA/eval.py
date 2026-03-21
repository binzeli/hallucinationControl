import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from scipy.stats import gaussian_kde, binom
from typing import Tuple, Dict
import glob
import re
import argparse

# Increase default font sizes for readability across all plots
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 22,
    'axes.labelsize': 16,
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14
})



def plot_idk_confidence_separate(df, file_label, reward_setting, output_dir):
    """
    Plots separate histograms for Answered (IDK=0) and Said IDK/Best Guess (IDK=1).
    Each plot shows correct vs incorrect breakdown as stacked bars.
    For baseline data (no idk_flag), plots a single histogram of all data.
    
    Optimized for conference paper publication with professional styling.
    
    Args:
        df: DataFrame with experiment results
        file_label: Label for the plot title
        reward_setting: Reward configuration string
        output_dir: Directory path where plots will be saved
    
    Returns:
        str: Path to the saved plot file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Sanitize file_label and reward_setting so they can't create subdirectories
    import re
    safe_label = re.sub(r'[^A-Za-z0-9_.-]+', '_', file_label)
    safe_reward = re.sub(r'[^A-Za-z0-9_.+-]+', '_', str(reward_setting))
    output_name = f"{safe_label}_({safe_reward})_{timestamp}.png"
    output_path = os.path.join(output_dir, output_name)

    # Set publication-quality style
    plt.rcParams['font.family'] = 'sans-serif'
    # Use DejaVu Sans first to avoid missing font warnings on clusters
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['axes.linewidth'] = 1
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['ytick.major.width'] = 1

    # Strip "Scheme " prefix from the display label used in plot titles
    display_label = file_label[7:] if file_label.lower().startswith('scheme ') else file_label

    df_plot = df.copy()
    
    # Check if this is baseline data (no idk_flag column or 'Baseline' in label)
    is_baseline = 'Baseline' in file_label
    
    if not is_baseline:
        df_plot.loc[df_plot['idk_flag'] == 0, 'confidence'] = df_plot.loc[df_plot['idk_flag'] == 0, 'first_confidence']
        df_plot.loc[df_plot['idk_flag'] == 1, 'confidence'] = df_plot.loc[df_plot['idk_flag'] == 1, 'best_guess_confidence']
    
    idk1_label = "Best Guess (IDK)"

    # Clean data
    df_plot['confidence'] = pd.to_numeric(df_plot['confidence'], errors='coerce')
    df_plot = df_plot.dropna(subset=['confidence'])
    df_plot['confidence'] = df_plot['confidence'].clip(0, 1)

    # Define professional color palette
    color_correct = '#4CAF50'  # Bright professional green
    color_incorrect = '#EF5350'  # Bright professional red
    color_correct_light = '#66BB6A'  # Lighter green
    color_incorrect_light = '#FF7675'  # Lighter red

    # Handle baseline case - single plot
    if is_baseline:
        bins = np.linspace(0, 1, 21)
        bin_edges = list(bins)
        total_count = len(df_plot)
        
        fig, ax = plt.subplots(figsize=(6.5, 3))
        
        # Get correct and incorrect data
        correct_data = df_plot.loc[df_plot['correct'] == 1, 'confidence']
        incorrect_data = df_plot.loc[df_plot['correct'] == 0, 'confidence']
        
        if len(correct_data) > 0 or len(incorrect_data) > 0:
            hist_correct, _ = np.histogram(correct_data, bins=bin_edges)
            hist_incorrect, _ = np.histogram(incorrect_data, bins=bin_edges)
            
            # Normalize by total dataset count
            hist_correct_norm = hist_correct / total_count
            hist_incorrect_norm = hist_incorrect / total_count
            
            bin_centers = [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]
            width = 0.04
            
            ax.bar(bin_centers, hist_correct_norm, width=width,
                   label='Correct', color=color_correct, alpha=0.85, edgecolor='white', linewidth=0.5)
            ax.bar(bin_centers, hist_incorrect_norm, width=width, bottom=hist_correct_norm,
                   label='Incorrect', color=color_incorrect, alpha=0.85, edgecolor='white', linewidth=0.5)
        
        ax.set_title(f'Confidence Distribution ({display_label}; {reward_setting})', fontsize=14, fontweight='bold', pad=12)
        ax.set_xlabel('Confidence', fontweight='bold', fontsize=14)
        ax.set_ylabel('Proportion of Dataset', fontweight='bold', fontsize=14)
        ax.set_ylim(0, 1)
        ax.legend(frameon=True, fancybox=False, shadow=False, edgecolor='black', framealpha=0.95, loc='upper right')
        ax.set_xticks(np.linspace(0, 1, 11))
        ax.set_xticklabels([f'{x:.1f}' for x in np.linspace(0, 1, 11)])
        ax.grid(axis='y', linestyle='-', alpha=0.2, linewidth=0.5, color='gray')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Saved baseline histogram: {output_path}")
        return output_path

    # Create bins for 0-1 range (20 bins of 0.05 each)
    bins = np.linspace(0, 1, 21)
    bin_edges = list(bins)
    total_count = len(df_plot)
    
    # Create two subplots with better spacing
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3), sharey=True)
    
    # ---- Plot 1: Answered (IDK=0) ----
    idk_0 = df_plot[df_plot['idk_flag'] == 0]
    if len(idk_0) > 0:
        idk_0_correct = idk_0.loc[idk_0['correct'] == 1, 'confidence']
        idk_0_incorrect = idk_0.loc[idk_0['correct'] == 0, 'confidence']
        
        hist_0_correct, _ = np.histogram(idk_0_correct, bins=bin_edges)
        hist_0_incorrect, _ = np.histogram(idk_0_incorrect, bins=bin_edges)
        
        # Normalize by total dataset count
        hist_0_correct_norm = hist_0_correct / total_count
        hist_0_incorrect_norm = hist_0_incorrect / total_count
        
        bin_centers = [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]
        width = 0.04
        
        ax1.bar(bin_centers, hist_0_correct_norm, width=width,
               label='Correct', color=color_correct, alpha=0.85, edgecolor='white', linewidth=0.5)
        ax1.bar(bin_centers, hist_0_incorrect_norm, width=width, bottom=hist_0_correct_norm,
               label='Incorrect', color=color_incorrect, alpha=0.85, edgecolor='white', linewidth=0.5)
        
        ax1.set_title(f'Answered ({display_label}; {reward_setting})', fontsize=14, fontweight='bold', pad=12)
        ax1.set_xlabel('Confidence', fontweight='bold', fontsize=14)
        ax1.set_ylabel('Proportion of Dataset', fontweight='bold', fontsize=14)
        ax1.set_ylim(0, 1)
        ax1.set_xticks(np.linspace(0, 1, 11))
        ax1.set_xticklabels([f'{x:.1f}' for x in np.linspace(0, 1, 11)])
        ax1.grid(axis='y', linestyle='-', alpha=0.2, linewidth=0.5, color='gray')
        ax1.set_axisbelow(True)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
    else:
        ax1.text(0.5, 0.5, 'No data', ha='center', va='center', 
                transform=ax1.transAxes, fontsize=11, style='italic', color='gray')
        ax1.set_title('Direct Answer', fontsize=12, fontweight='bold', pad=12)
        ax1.set_xlabel('Confidence', fontweight='bold')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
    
    # ---- Plot 2: Said IDK / Best Guess (IDK=1) ----
    idk_1 = df_plot[df_plot['idk_flag'] == 1]
    if len(idk_1) > 0:
        idk_1_correct = idk_1.loc[idk_1['correct'] == 1, 'confidence']
        idk_1_incorrect = idk_1.loc[idk_1['correct'] == 0, 'confidence']
        
        hist_1_correct, _ = np.histogram(idk_1_correct, bins=bin_edges)
        hist_1_incorrect, _ = np.histogram(idk_1_incorrect, bins=bin_edges)
        
        # Normalize by total dataset count
        hist_1_correct_norm = hist_1_correct / total_count
        hist_1_incorrect_norm = hist_1_incorrect / total_count
        
        bin_centers = [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]
        width = 0.04
        
        ax2.bar(bin_centers, hist_1_correct_norm, width=width,
               label='Correct', color=color_correct_light, alpha=0.85, edgecolor='white', linewidth=0.5)
        ax2.bar(bin_centers, hist_1_incorrect_norm, width=width, bottom=hist_1_correct_norm,
               label='Incorrect', color=color_incorrect_light, alpha=0.85, edgecolor='white', linewidth=0.5)
        
        ax2.set_title(f'Best Guess (after saying IDK) ({display_label}; {reward_setting})', fontsize=14, fontweight='bold', pad=12)
        ax2.set_xlabel('Confidence', fontweight='bold', fontsize=14)
        ax2.set_xticks(np.linspace(0, 1, 11))
        ax2.set_xticklabels([f'{x:.1f}' for x in np.linspace(0, 1, 11)])
        ax2.grid(axis='y', linestyle='-', alpha=0.2, linewidth=0.5, color='gray')
        ax2.set_axisbelow(True)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
    else:
        ax2.text(0.5, 0.5, 'No data', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=11, style='italic', color='gray')
        ax2.set_title('Best Guess (IDK)', fontsize=12, fontweight='bold', pad=12)
        ax2.set_xlabel('Confidence', fontweight='bold')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
    
    # Add legend to top right of the second subplot
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels, loc='upper right', frameon=True, fancybox=False, 
              shadow=False, edgecolor='black', framealpha=0.95)
    
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Saved: {output_path}")
    return output_path


def calculate_calibration_error(df: pd.DataFrame, file_label: str, num_bins: int = 10) -> Dict[str, float]:
    """
    Calculate Expected Calibration Error (ECE).
    
    ECE = (1/m) * sum over bins: m_b * |o_hat_b - p_hat_b|
    
    where:
    - m is total number of samples
    - m_b is number of samples in bin b
    - o_hat_b is empirical frequency (accuracy) in bin b
    - p_hat_b is average predicted probability (self reported confidence) in bin b
    
    For file_label 'B', uses best_guess_confidence for IDK=1;
    otherwise uses first_confidence for both.
    """
    
    df_calc = df.copy()
    if file_label != "A_Baseline":
        df_calc.loc[df_calc['idk_flag'] == 0, 'confidence'] = df_calc.loc[df_calc['idk_flag'] == 0, 'first_confidence']
        df_calc.loc[df_calc['idk_flag'] == 1, 'confidence'] = df_calc.loc[df_calc['idk_flag'] == 1, 'best_guess_confidence']
    
    # Clean data
    df_calc['confidence'] = pd.to_numeric(df_calc['confidence'], errors='coerce')
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
    Calculate calibration error separately for answered questions (idk_flag=0)
    and all questions combined.
    
    Returns:
        Dictionary with separate calibration metrics for each category
    """
    results = {}
    if file_label == "A_Baseline":
        df = df[df['idk_flag'] == 0].copy()
        results['overall'] = calculate_calibration_error(df, file_label, num_bins)
        return results

    # Split by IDK flag
    df_answered = df[df['idk_flag'] == 0].copy()
    
    # Calculate for answered questions
    if len(df_answered) > 0:
        results['answered'] = calculate_calibration_error(df_answered, file_label, num_bins)
    else:
        results['answered'] = None
    
    # Calculate for all combined
    results['overall'] = calculate_calibration_error(df, file_label, num_bins)
    
    return results


def print_calibration_report(results: Dict[str, Dict], label: str):
    """
    Pretty print calibration error results.
    
    Args:
        results: Dictionary returned by calculate_calibration_by_category
        label: Label for the experiment (e.g., 'Experiment A')
    """
    
    print(f"\n{'='*80}")
    print(f"📏 CALIBRATION REPORT: {label}")
    print(f"{'='*80}")
    
    for category, metrics in results.items():
        if metrics is None:
            print(f"\n❌ {category.upper()}: No data available")
            continue
        
        print(f"\n📊 {category.upper()}")
        print(f"   Expected Calibration Error (ECE): {metrics['ECE']:.4f}")
        print(f"   Total samples: {metrics['total_samples']}")
        print(f"   Number of bins: {metrics['num_bins']}")
        
        print(f"\n   Per-bin breakdown:")
        print(f"   {'Bin':<15} {'Count':<8} {'P_hat':<12} {'O_hat':<12} {'Cal Gap':<12}")
        print(f"   {'-'*60}")
        
        for bin_stat in metrics['bin_stats']:
            if bin_stat['count'] > 0:
                print(f"   {bin_stat['bin_range']:<15} "
                      f"{bin_stat['count']:<8} "
                      f"{bin_stat['avg_predicted_prob']:<12.4f} "
                      f"{bin_stat['empirical_frequency']:<12.4f} "
                      f"{bin_stat['calibration_gap']:<12.4f}")




def calculate_brier_score(df: pd.DataFrame) -> float:
    """
    Calculate the Brier score for the dataset.

    Brier score = mean((confidence - correctness)^2)
    """
    df['confidence'] = df.apply(
        lambda row: row['first_confidence'] if row['idk_flag'] == 0 else row['best_guess_confidence'], axis=1
    )
    df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')
    df['confidence'] = df['confidence'].clip(0, 1)

    brier_score = np.mean((df['confidence'] - df['correct']) ** 2)
    return brier_score


def calculate_brier_score_with_ci(df: pd.DataFrame, confidence: float = 0.95, answered_only: bool = False, file_label = None) -> Tuple[float, float]:
    """
    Calculate the Brier score and its confidence interval for the dataset.

    Args:
        df: DataFrame containing the data.
        confidence: Confidence level for the interval (default: 95%).
        file_label: Optional label for the file (not used in calculation).
    Returns:
        A tuple containing the Brier score and the margin of error (Brier score ± margin).
    """
    # If requested, restrict to answered rows only and use their first_confidence.

    if file_label != "A_Baseline":
        if answered_only:
            df = df[df['idk_flag'] == 0].copy()
            df['confidence'] = pd.to_numeric(df['first_confidence'], errors='coerce')
        else:
            df['confidence'] = df.apply(
            lambda row: row['first_confidence'] if row['idk_flag'] == 0 else row['best_guess_confidence'], axis=1
            )
    else:
        df = df[df['idk_flag'] == 0].copy()
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

def summarize(df, label):

        acc = df["correct"].mean()
        answered = df["idk_flag"] == 0 
        far_e1 = df.loc[answered, "false_answer_flag"].sum() / answered.sum()
        far_e2 = 1 - acc
        print(f"\n📊 {label} Summary:")


        far_e1_margin = calculate_ci_pm(df.loc[answered, "false_answer_flag"].sum(), answered.sum())
        far_e2_margin = calculate_ci_pm(len(df) - df["correct"].sum(), len(df))
        
        print(f"  False-Answer Rate - Answered: {far_e1:.3f} ± {far_e1_margin:.3f}")
        print(f"  False-Answer Rate - Overall: {far_e2:.3f} ± {far_e2_margin:.3f}")
       
        # calculate total score
        total_score = df["score"].sum()
        print(f"  Total Score: {total_score} out of {len(df)}")

        answered = df["idk_flag"] == 0
        wrong_answered = df[answered & (df["correct"] == 0)]
        print(f"  Answered questions: {answered.sum()} out of {len(df)}")
        print(f"  Wrong answered questions: {len(wrong_answered)} out of {answered.sum()}")
        
        # calculate number of abstain question out of wrong answered questions
        wrong_answered = df[df["correct"] == 0]
        print(f"  Total wrong answered questions: {len(wrong_answered)}")
        abstain_out_of_wrong = (len(wrong_answered[wrong_answered["idk_flag"] == 1]) / len(wrong_answered)) if len(wrong_answered) > 0 else 0
        print(f"  Abstain out of wrong answered questions: {abstain_out_of_wrong:.3f}")

     



def parse_filename(filename):
    """
    Parse experiment filename to extract model name, scheme and reward values.
    Example: main_exp/outputs/popqa/gpt5_results/popqa_scheme_b_norm_results_+1_+0_+0.4_20260308_221816.csv
    Returns: (model_name, scheme_name, reward_correct, reward_incorrect, reward_abstain)
    """

    basename = os.path.basename(filename)
    
    # Extract model name from directory path (e.g., gpt5_results -> gpt5)
    model_match = re.search(r'/([^/]+)_results/', filename)
    model = model_match.group(1) if model_match else "gpt5"
    
    # Extract scheme name (between popqa_ and _results)
    scheme_match = re.search(r'popqa_(.+?)_results', basename)
    scheme = scheme_match.group(1) if scheme_match else "unknown"
    
    # Extract reward values (format: +1_-1_+0.4 or +1_0_+0.4)
    # Rewards are between _results_ and timestamp (8 digits)
    reward_match = re.search(r'_results_([+-]?\d+(?:\.\d+)?)_([+-]?\d+(?:\.\d+)?)_([+-]?\d+(?:\.\d+)?)_\d{8}', basename)
    if reward_match:
        reward_correct = reward_match.group(1)
        reward_incorrect = reward_match.group(2)
        reward_abstain = reward_match.group(3)
    else:
        reward_correct = "+1"
        reward_incorrect = "-1"
        reward_abstain = "+0.4"
    
    return model, scheme, reward_correct, reward_incorrect, reward_abstain


def get_latest_file(pattern):
    """Find the most recent file matching the pattern."""

    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {pattern}")
    # Sort by modification time, most recent first
    latest_file = max(files, key=os.path.getmtime)
    return latest_file


def get_all_result_files(results_dir="main_exp/outputs/popqa/gpt5_results"):
    """
    Find all result CSV files in the specified directory.
    Returns a dictionary mapping scheme names to their file paths.
    """
    
    pattern = os.path.join(results_dir, "popqa_*_results_*.csv")
    files = glob.glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"No result files found in {results_dir}")
    
    # Group files by scheme and keep only the latest for each scheme
    scheme_files = {}
    for file in files:
        model, scheme, _, _, _ = parse_filename(file)
        if scheme not in scheme_files:
            scheme_files[scheme] = file
        else:
            # Keep the more recent file
            if os.path.getmtime(file) > os.path.getmtime(scheme_files[scheme]):
                scheme_files[scheme] = file
    
    return scheme_files


def main():

    
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Evaluate experiment results from CSV file.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--result-file", "-f",
        type=str,
        required=True,
        help="Path to the result CSV file to analyze"
    )
    
    args = parser.parse_args()
    
    # Get the result file path
    filepath = args.result_file
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"❌ Error: File not found: {filepath}")
        return
    
    print(f"\n📊 Analyzing result file: {filepath}")
    
    # Parse filename to get model name, scheme and reward settings
    model_name, scheme_name, r_correct, r_incorrect, r_abstain = parse_filename(filepath)
    
    # Build reward setting string (exclude abstain reward for scheme_a)
    reward_parts = [r_correct, r_incorrect]
    if scheme_name.lower().startswith("scheme_b"):
        reward_parts.append(r_abstain)
    reward_setting = ", ".join(reward_parts)
    
    print(f"\n{'='*80}")
    print(f"Processing: {scheme_name}")
    print(f"{'='*80}")
    print(f"  Model: {model_name}")
    print(f"  Scheme: {scheme_name}")
    print(f"  Rewards: {reward_setting}")
    print(f"  File: {filepath}")
    
    # Load data
    df = pd.read_csv(filepath)
    
    # Determine file label for analysis functions
    if 'baseline' in scheme_name.lower():
        file_label = f"{scheme_name.replace('_', ' ').title()}"
        analysis_label = "A_Baseline"
    else:
        label = scheme_name.lower()

        if "b_norm" in label:
            file_label = "Scheme B w/ norms"
        elif "scheme_b" in label:
            file_label = "Scheme B"
        elif "scheme_a" in label:
            file_label = "Scheme A"
        else:
            file_label = scheme_name.replace('_', ' ').title()
    
        analysis_label = scheme_name.upper().replace('SCHEME_', '')
    
    # Track all output files
    output_files = []
    
    # Setup output directory with parsed model name
    output_dir = f"main_exp/popQA/outputs/{model_name}_results/plot_idk_confidence_separate"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run summary
    summarize(df, f"{file_label}")
    
    # Generate plots 
    plot_file = plot_idk_confidence_separate(df, file_label, reward_setting, output_dir)
    if plot_file:
        output_files.append(plot_file)
    
    ######################### ECE calculation #########################
    results = calculate_calibration_by_category(df, analysis_label, num_bins=10)
    print_calibration_report(results, f"{file_label}")
    
    ######################### Brier Score calculation #########################
    brier_score, margin = calculate_brier_score_with_ci(df)
    print(f"\nBrier Score (overall): {brier_score:.4f} ± {margin:.4f}")
    
    # Only calculate answered-only Brier if not baseline
    if 'baseline' not in scheme_name.lower():
        brier_score_answered, margin_answered = calculate_brier_score_with_ci(df, answered_only=True)
        print(f"Brier Score (answered only): {brier_score_answered:.4f} ± {margin_answered:.4f}")




if __name__ == "__main__":
    main()
