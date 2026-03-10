"""
Analysis script for PopQA experiment results (No Reward version, single run, no repeats).

This script calculates the same metrics as data_analysis.py, calculate_ece.py, and calculate_brier_score.py,
but adapted for single-run results (no repeated experiments, no median aggregation needed).

Input CSV format (from run_PopQA.py):
dataset_id, timestamp, s_pop, o_pop, question, possible_answers, 
first_answer, first_confidence, best_guess, best_guess_confidence,
correct, score, idk_flag,
self_confidence, api_confidence_min, api_confidence_avg

Note: self_confidence, api_confidence_min, and api_confidence_avg are already processed
based on IDK logic (if IDK, use best_guess values; otherwise use first_answer values).
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy import stats
import ast


def load_results(file_path):
    """Load PopQA experiment results from a CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    data = pd.read_csv(file_path)
    
    # Ensure correct column exists (correct is 0/1)
    if 'correct' not in data.columns:
        raise ValueError("Column 'correct' not found in the CSV file.")
    
    # Ensure confidence columns exist
    if 'self_confidence' not in data.columns:
        raise ValueError("Column 'self_confidence' not found in the CSV file.")
    
    return data


def calculate_ece_global(df: pd.DataFrame, confidence_col: str, num_bins: int = 10) -> float:
    """
    Calculate ECE on the entire dataset using standard formula:
    ECÊ = (1/m) * Σ_{b=1}^B m_b * |ô_b - p̂_b|
    
    where:
    - m is total number of questions
    - m_b is number of questions in bin b
    - ô_b is empirical frequency (accuracy) in bin b
    - p̂_b is average predicted probability (confidence) in bin b
    
    Args:
        df: DataFrame with confidence and correct columns
        confidence_col: Name of the confidence column
        num_bins: Number of bins for ECE calculation
    
    Returns:
        Global ECE value
    """
    if len(df) == 0:
        return np.nan
    
    # Clean confidence values
    df_clean = df[[confidence_col, 'correct']].copy()
    df_clean[confidence_col] = pd.to_numeric(df_clean[confidence_col], errors='coerce')
    df_clean[confidence_col] = df_clean[confidence_col].clip(0, 1)
    df_clean = df_clean.dropna(subset=[confidence_col, 'correct'])
    
    if len(df_clean) == 0:
        return np.nan
    
    # Create bins: partition [0, 1] into B bins
    bins = np.linspace(0, 1, num_bins + 1)
    df_clean['bin'] = pd.cut(df_clean[confidence_col], bins=bins, include_lowest=True, labels=False)
    
    # Calculate ECE using standard formula
    ece = 0.0
    total_samples = len(df_clean)  # m in the formula
    
    for bin_idx in range(num_bins):
        bin_data = df_clean[df_clean['bin'] == bin_idx]
        
        if len(bin_data) > 0:
            bin_count = len(bin_data)  # m_b in the formula
            
            # p̂_b: average predicted probability in bin b
            bin_avg_confidence = bin_data[confidence_col].mean()
            
            # ô_b: empirical frequency (accuracy) in bin b
            bin_empirical_frequency = bin_data['correct'].mean()
            
            # Calibration gap: |ô_b - p̂_b|
            calibration_gap = abs(bin_empirical_frequency - bin_avg_confidence)
            
            # Weighted contribution: (m_b / m) * |ô_b - p̂_b|
            ece += (bin_count / total_samples) * calibration_gap
    
    return ece


def calculate_brier_score_single(correct: int, confidence: float) -> float:
    """
    Calculate Brier score for a single record.
    
    Args:
        correct: 1 if correct, 0 if incorrect
        confidence: The confidence score (0-1)
    
    Returns:
        Brier score: (actual_outcome - confidence)^2
    """
    # actual_outcome is already 0 or 1 (from correct column)
    actual_outcome = float(correct)
    
    # Calculate Brier score: (actual_outcome - confidence)^2
    brier_score = (actual_outcome - confidence) ** 2
    
    return brier_score


def calculate_pearson_coefficient(data, api_conf_col: str):
    """
    Calculate the Pearson correlation coefficient with 95% confidence interval.
    Calculates correlation between Self Confidence and API Confidence.
    
    Uses Fisher z-transformation to calculate confidence interval for correlation.
    
    Args:
        data: DataFrame with self_confidence and api_conf_col columns
        api_conf_col: Name of the API confidence column
    
    Returns:
        (pearson_r, pearson_p, (ci_lower, ci_upper), spearman_r, spearman_p, kendall_r, kendall_p)
    """
    if 'self_confidence' not in data.columns:
        print(f"Warning: Column self_confidence not found")
        return np.nan, np.nan, (np.nan, np.nan), np.nan, np.nan, np.nan, np.nan
    
    if api_conf_col not in data.columns:
        print(f"Warning: Column {api_conf_col} not found")
        return np.nan, np.nan, (np.nan, np.nan), np.nan, np.nan, np.nan, np.nan
    
    # Get valid data (drop rows where either value is NaN)
    valid_data = data[[api_conf_col, "self_confidence"]].dropna()
    
    if len(valid_data) < 3:
        print(f"Insufficient data for correlation calculation (need at least 3 points)")
        return np.nan, np.nan, (np.nan, np.nan), np.nan, np.nan, np.nan, np.nan
    
    # Check for constant values (which would cause correlation to be undefined)
    api_values = valid_data[api_conf_col].values
    self_values = valid_data["self_confidence"].values
    
    if np.std(api_values) == 0 or np.std(self_values) == 0:
        return np.nan, np.nan, (np.nan, np.nan), np.nan, np.nan, np.nan, np.nan
    
    # Calculate correlation
    correlation, p_value = pearsonr(api_values, self_values)
    n = len(valid_data)
    
    # Check if correlation is valid
    if np.isnan(correlation):
        return np.nan, np.nan, (np.nan, np.nan), np.nan, np.nan, np.nan, np.nan
    
    # Calculate 95% confidence interval using Fisher z-transformation
    if abs(correlation) < 0.9999:  # Avoid division by zero for perfect correlation
        # Fisher z-transformation
        z = 0.5 * np.log((1 + correlation) / (1 - correlation))
        
        # Standard error of z
        se_z = 1 / np.sqrt(n - 3)
        
        # 95% CI for z (using normal distribution, not t-distribution for correlation)
        z_critical = stats.norm.ppf(0.975)  # 1.96 for 95% CI
        z_lower = z - z_critical * se_z
        z_upper = z + z_critical * se_z
        
        # Transform back to correlation scale (inverse Fisher transform)
        ci_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
        ci_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
    else:
        # For perfect or near-perfect correlation, CI is not well-defined
        ci_lower = correlation
        ci_upper = correlation
    
    # Calculate Spearman correlation (monotonic relationship, rank-based)
    try:
        spearman_corr, spearman_p = spearmanr(api_values, self_values)
        if np.isnan(spearman_corr):
            spearman_corr, spearman_p = np.nan, np.nan
        else:
            # Approximate CI for Spearman using Fisher z-transformation
            if abs(spearman_corr) < 0.9999:
                z_spearman = 0.5 * np.log((1 + spearman_corr) / (1 - spearman_corr))
                se_z_spearman = 1.06 / np.sqrt(n - 3)
                z_critical = stats.norm.ppf(0.975)
                z_lower_spearman = z_spearman - z_critical * se_z_spearman
                z_upper_spearman = z_spearman + z_critical * se_z_spearman
                ci_lower_spearman = (np.exp(2 * z_lower_spearman) - 1) / (np.exp(2 * z_lower_spearman) + 1)
                ci_upper_spearman = (np.exp(2 * z_upper_spearman) - 1) / (np.exp(2 * z_upper_spearman) + 1)
            else:
                ci_lower_spearman = spearman_corr
                ci_upper_spearman = spearman_corr
    except Exception as e:
        spearman_corr, spearman_p = np.nan, np.nan
        ci_lower_spearman, ci_upper_spearman = np.nan, np.nan
    
    # Calculate Kendall's tau
    try:
        kendall_tau, kendall_p = kendalltau(api_values, self_values)
        if np.isnan(kendall_tau):
            kendall_tau, kendall_p = np.nan, np.nan
    except Exception:
        kendall_tau, kendall_p = np.nan, np.nan
    
    return correlation, p_value, (ci_lower, ci_upper), spearman_corr, spearman_p, kendall_tau, kendall_p


def process_csv_file(input_file_path: str, output_dir: str, num_bins: int = 10):
    """
    Process a PopQA CSV file to calculate all metrics and generate visualizations.
    
    Args:
        input_file_path: Path to input CSV file
        output_dir: Directory to save output visualizations
        num_bins: Number of bins for ECE calculation
    """
    # Load data
    data = load_results(input_file_path)
    print(f"\n{'='*80}")
    print(f"Processing: {os.path.basename(input_file_path)}")
    print(f"{'='*80}")
    print(f"Total questions: {len(data)}")
    
    # Print additional statistics if available
    if 'idk_flag' in data.columns:
        idk_count = data['idk_flag'].sum()
        idk_pct = 100 * idk_count / len(data)
        print(f"IDK responses: {idk_count} ({idk_pct:.2f}%)")
        print(f"Direct answers: {len(data) - idk_count} ({100 - idk_pct:.2f}%)")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # ============================================================================
    # 1. Calculate ECE
    # ============================================================================
    print(f"\n{'='*80}")
    print("Expected Calibration Error (ECE)")
    print(f"{'='*80}")
    
    # ECE for self_confidence
    if 'self_confidence' in data.columns:
        ece_self = calculate_ece_global(data, 'self_confidence', num_bins=num_bins)
        print(f"\nSelf-Confidence ECE: {ece_self:.6f}")
        print(f"Number of questions: {len(data[['self_confidence', 'correct']].dropna())}")
    
    # ECE for API confidence methods
    api_methods = [
        ('api_confidence_min', 'API Confidence (min)'),
        ('api_confidence_avg', 'API Confidence (average)'),
    ]
    
    for api_col, label in api_methods:
        if api_col in data.columns:
            ece_api = calculate_ece_global(data, api_col, num_bins=num_bins)
            print(f"\n{label} ECE: {ece_api:.6f}")
            print(f"Number of questions: {len(data[[api_col, 'correct']].dropna())}")
    
    # ============================================================================
    # 2. Calculate Brier Score
    # ============================================================================
    print(f"\n{'='*80}")
    print("Brier Score")
    print(f"{'='*80}")
    
    # Brier score for self_confidence
    if 'self_confidence' in data.columns and 'correct' in data.columns:
        data['self_brier_score'] = data.apply(
            lambda row: calculate_brier_score_single(row['correct'], row['self_confidence']) if pd.notna(row['self_confidence']) else np.nan,
            axis=1
        )
        
        brier_scores = data['self_brier_score'].dropna().values
        if len(brier_scores) > 0:
            avg_brier_score = np.mean(brier_scores)
            variance = np.var(brier_scores, ddof=1)
            std_dev = np.std(brier_scores, ddof=1)
            n = len(brier_scores)
            se = std_dev / np.sqrt(n)
            t_critical = stats.t.ppf(0.975, df=n-1)
            ci_lower = avg_brier_score - t_critical * se
            ci_upper = avg_brier_score + t_critical * se
            ci_margin = t_critical * se
            
            print(f"\nSelf-Confidence Brier Score:")
            print(f"  Average: {avg_brier_score:.6f}")
            print(f"  Summary: {avg_brier_score:.4f}(± {ci_margin:.4f})")
            print(f"  Variance: {variance:.6f}")
            print(f"  Standard Deviation: {std_dev:.6f}")
            print(f"  95% Confidence Interval: [{ci_lower:.6f}, {ci_upper:.6f}]")
            print(f"  Number of records: {n}")
    
    # Brier score for API confidence methods (same api_methods as ECE)
    for api_col, label in api_methods:
        if api_col in data.columns and 'correct' in data.columns:
            brier_col = f'{api_col}_brier_score'
            data[brier_col] = data.apply(
                lambda row: calculate_brier_score_single(row['correct'], row[api_col]) if pd.notna(row[api_col]) else np.nan,
                axis=1
            )
            
            brier_scores = data[brier_col].dropna().values
            if len(brier_scores) > 0:
                avg_brier_score = np.mean(brier_scores)
                variance = np.var(brier_scores, ddof=1)
                std_dev = np.std(brier_scores, ddof=1)
                n = len(brier_scores)
                se = std_dev / np.sqrt(n)
                t_critical = stats.t.ppf(0.975, df=n-1)
                ci_lower = avg_brier_score - t_critical * se
                ci_upper = avg_brier_score + t_critical * se
                ci_margin = t_critical * se
                
                print(f"\n{label} Brier Score:")
                print(f"  Average: {avg_brier_score:.6f}")
                print(f"  Summary: {avg_brier_score:.4f}(± {ci_margin:.4f})")
                print(f"  Variance: {variance:.6f}")
                print(f"  Standard Deviation: {std_dev:.6f}")
                print(f"  95% Confidence Interval: [{ci_lower:.6f}, {ci_upper:.6f}]")
                print(f"  Number of records: {n}")
    
    # ============================================================================
    # 3. Task performance: False Answer Rate ( = 1 - accuracy )
    # ============================================================================
    print(f"\n{'='*80}")
    print("Task Performance: False Answer Rate")
    print(f"{'='*80}")
    
    if 'correct' in data.columns:
        correct_values = data['correct'].dropna().values
        if len(correct_values) > 0:
            avg_accuracy = np.mean(correct_values)
            std_dev_accuracy = np.std(correct_values, ddof=1)
            n_accuracy = len(correct_values)
            se_accuracy = std_dev_accuracy / np.sqrt(n_accuracy)
            t_critical_accuracy = stats.t.ppf(0.975, df=n_accuracy - 1)
            ci_lower_accuracy = avg_accuracy - t_critical_accuracy * se_accuracy
            ci_upper_accuracy = avg_accuracy + t_critical_accuracy * se_accuracy
            ci_margin_accuracy = t_critical_accuracy * se_accuracy

            # False answer rate = 1 - accuracy; CI for false rate = [1 - acc_upper, 1 - acc_lower]
            false_answer_rate = 1.0 - avg_accuracy
            ci_lower_far = 1.0 - ci_upper_accuracy
            ci_upper_far = 1.0 - ci_lower_accuracy
            ci_margin_far = ci_margin_accuracy  # same half-width

            print(f"False Answer Rate: {false_answer_rate:.6f}")
            print(f"Summary: {false_answer_rate:.4f}(± {ci_margin_far:.4f})")
            print(f"95% Confidence Interval: [{ci_lower_far:.6f}, {ci_upper_far:.6f}]")
            print(f"Number of questions: {n_accuracy}")
    
    # ============================================================================
    # 4. Calculate Correlation
    # ============================================================================
    print(f"\n{'='*80}")
    print("Correlation: Self Confidence vs API Confidence")
    print(f"{'='*80}")
    
    correlation_results = []
    
    for api_col, label in api_methods:
        if api_col in data.columns:
            pearson_r, pearson_p, pearson_ci, spearman_r, spearman_p, kendall_r, kendall_p = calculate_pearson_coefficient(data, api_col)
            
            correlation_results.append({
                'method': label,
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'pearson_ci': pearson_ci,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'kendall_r': kendall_r,
                'kendall_p': kendall_p
            })
    
    # Print correlation summary table
    print(f"\n{'Method':<45} {'Pearson Correlation (95% CI)':<35}")
    print(f"{'-'*80}")
    
    for result in correlation_results:
        method = result['method']
        pearson_r = result['pearson_r']
        pearson_p = result['pearson_p']
        pearson_ci = result.get('pearson_ci', (np.nan, np.nan))
        
        # Format Pearson correlation with 95% CI as correlation(± margin)
        if np.isnan(pearson_r):
            corr_str = "NaN"
        else:
            ci_lower, ci_upper = pearson_ci
            if not (np.isnan(ci_lower) or np.isnan(ci_upper)):
                # Calculate margin (half the width of CI)
                margin = (ci_upper - ci_lower) / 2
                corr_str = f"{pearson_r:.4f}(± {margin:.4f})"
            else:
                corr_str = f"{pearson_r:.4f}"
        
        print(f"{method:<45} {corr_str:<35}")
    
    return data


if __name__ == "__main__":
    # Set input file name here
    input_file = "popQA-20260309_183536.csv"  # Change this to your input CSV file name
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "outputs")
    
    # Determine input file path
    if os.path.isabs(input_file) or os.path.exists(input_file):
        input_path = input_file
    else:
        input_path = os.path.join(results_dir, input_file)
    
    # Output directory for visualizations (same as results directory)
    output_dir = results_dir
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        print(f"Please update the input_file variable in the script.")
    else:
        try:
            process_csv_file(input_path, output_dir, num_bins=10)
        except Exception as e:
            print(f"Error processing file: {e}")
            import traceback
            traceback.print_exc()

