"""
Analysis script for MMLU-Pro experiment results.

Aligns with popQA_analysis_no_reward.py: same metrics (ECE, Brier, False Answer Rate, Pearson correlation)
but only compares self_confidence vs api_confidence_key_token (key-token log prob).

Input CSV format (from experiment_0_v3.0.py or compatible):
  index, question, options, correct_answer,
  model_answers, model_answer, answer_distribution,
  self_confidences, api_confidences_key_token

Accuracy/Brier: question-level average over repeats, then average of these (matches calculate_brier_score.py).
ECE/Correlation: median of confidences and median of correctness per question.
"""

import argparse
import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy import stats


def parse_confidence_string(conf_str):
    """Parse semicolon-separated confidence string and return list of floats."""
    if pd.isna(conf_str) or conf_str == '':
        return []
    return [float(x.strip()) for x in str(conf_str).split(';') if x.strip()]


def calculate_question_accuracy(correct_answer, model_answers_str):
    """
    Per-question accuracy: correct_count / total_repeats.
    Matches calculate_brier_score.calculate_question_accuracy.
    """
    if pd.isna(model_answers_str) or model_answers_str == '':
        return np.nan
    model_answers = [ans.strip() for ans in str(model_answers_str).split(';') if ans.strip()]
    if len(model_answers) == 0:
        return np.nan
    correct_count = sum(1 for ans in model_answers if ans == correct_answer)
    return correct_count / len(model_answers)


def calculate_average_brier_score(correct_answer, model_answers_str, confidences_str):
    """
    Per-question average Brier score across repeats.
    Matches calculate_brier_score.calculate_average_brier_score.
    """
    model_answers = [ans.strip() for ans in str(model_answers_str).split(';') if ans.strip()]
    confidences = parse_confidence_string(confidences_str)
    min_len = min(len(model_answers), len(confidences))
    model_answers = model_answers[:min_len]
    confidences = confidences[:min_len]
    if min_len == 0:
        return np.nan
    brier_scores = [
        ((1 if ma == correct_answer else 0) - conf) ** 2
        for ma, conf in zip(model_answers, confidences)
    ]
    return np.mean(brier_scores)


def load_raw_data(file_path):
    """Load raw MMLU-Pro CSV (no aggregation)."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    return pd.read_csv(file_path)


def load_and_aggregate_results(file_path):
    """
    Load MMLU-Pro CSV and aggregate repeats to median per question (for ECE and correlation).
    
    Returns DataFrame with columns: index, question, options, correct_answer, model_answer,
    self_confidence, api_confidence_key_token, correct (median for ECE binning)
    """
    data = load_raw_data(file_path)
    
    if 'self_confidences' not in data.columns:
        raise ValueError("Column 'self_confidences' not found in the CSV file.")
    if 'api_confidences_key_token' not in data.columns:
        raise ValueError("Column 'api_confidences_key_token' not found in the CSV file.")
    
    aggregated = []
    for idx, row in data.iterrows():
        model_answers_str = row.get('model_answers', '')
        self_conf_str = row.get('self_confidences', '')
        api_conf_str = row.get('api_confidences_key_token', '')
        correct_answer = row.get('correct_answer', '')
        
        self_vals = parse_confidence_string(self_conf_str)
        api_vals = parse_confidence_string(api_conf_str)
        
        if not self_vals or not api_vals:
            continue
        
        self_confidence = np.median(self_vals)
        api_confidence_key_token = np.median(api_vals)
        
        # Median of correctness for ECE (majority vote)
        model_answers = [ans.strip() for ans in str(model_answers_str).split(';') if ans.strip()]
        if not model_answers:
            continue
        correctness_values = [1 if ans == correct_answer else 0 for ans in model_answers]
        median_correct = np.median(correctness_values)
        if median_correct == 0.5:
            median_correct = np.mean(correctness_values)
        correct = int(round(median_correct))
        
        aggregated.append({
            'index': row.get('index', idx),
            'question': row.get('question', ''),
            'options': row.get('options', ''),
            'correct_answer': correct_answer,
            'model_answer': row.get('model_answer', ''),
            'self_confidence': self_confidence,
            'api_confidence_key_token': api_confidence_key_token,
            'correct': correct,
        })
    
    return pd.DataFrame(aggregated)


def calculate_ece_global(df: pd.DataFrame, confidence_col: str, num_bins: int = 10) -> float:
    """Calculate ECE using standard binned formula."""
    if len(df) == 0:
        return np.nan
    
    df_clean = df[[confidence_col, 'correct']].copy()
    df_clean[confidence_col] = pd.to_numeric(df_clean[confidence_col], errors='coerce')
    df_clean[confidence_col] = df_clean[confidence_col].clip(0, 1)
    df_clean = df_clean.dropna(subset=[confidence_col, 'correct'])
    
    if len(df_clean) == 0:
        return np.nan
    
    bins = np.linspace(0, 1, num_bins + 1)
    df_clean['bin'] = pd.cut(df_clean[confidence_col], bins=bins, include_lowest=True, labels=False)
    
    ece = 0.0
    total_samples = len(df_clean)
    
    for bin_idx in range(num_bins):
        bin_data = df_clean[df_clean['bin'] == bin_idx]
        if len(bin_data) > 0:
            bin_count = len(bin_data)
            bin_avg_confidence = bin_data[confidence_col].mean()
            bin_empirical_frequency = bin_data['correct'].mean()
            calibration_gap = abs(bin_empirical_frequency - bin_avg_confidence)
            ece += (bin_count / total_samples) * calibration_gap
    
    return ece


def calculate_pearson_coefficient(data, api_conf_col: str):
    """Pearson correlation between self_confidence and api_conf_col with 95% CI."""
    if 'self_confidence' not in data.columns:
        return np.nan, np.nan, (np.nan, np.nan), np.nan, np.nan, np.nan, np.nan
    if api_conf_col not in data.columns:
        return np.nan, np.nan, (np.nan, np.nan), np.nan, np.nan, np.nan, np.nan
    
    valid_data = data[[api_conf_col, "self_confidence"]].dropna()
    if len(valid_data) < 3:
        return np.nan, np.nan, (np.nan, np.nan), np.nan, np.nan, np.nan, np.nan
    
    api_values = valid_data[api_conf_col].values
    self_values = valid_data["self_confidence"].values
    if np.std(api_values) == 0 or np.std(self_values) == 0:
        return np.nan, np.nan, (np.nan, np.nan), np.nan, np.nan, np.nan, np.nan
    
    correlation, p_value = pearsonr(api_values, self_values)
    n = len(valid_data)
    
    if np.isnan(correlation):
        return np.nan, np.nan, (np.nan, np.nan), np.nan, np.nan, np.nan, np.nan
    
    if abs(correlation) < 0.9999:
        z = 0.5 * np.log((1 + correlation) / (1 - correlation))
        se_z = 1 / np.sqrt(n - 3)
        z_critical = stats.norm.ppf(0.975)
        z_lower = z - z_critical * se_z
        z_upper = z + z_critical * se_z
        ci_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
        ci_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
    else:
        ci_lower = ci_upper = correlation
    
    try:
        spearman_corr, spearman_p = spearmanr(api_values, self_values)
        if np.isnan(spearman_corr):
            spearman_corr, spearman_p = np.nan, np.nan
    except Exception:
        spearman_corr, spearman_p = np.nan, np.nan
    
    try:
        kendall_tau, kendall_p = kendalltau(api_values, self_values)
        if np.isnan(kendall_tau):
            kendall_tau, kendall_p = np.nan, np.nan
    except Exception:
        kendall_tau, kendall_p = np.nan, np.nan
    
    return correlation, p_value, (ci_lower, ci_upper), spearman_corr, spearman_p, kendall_tau, kendall_p


def process_csv_file(input_file_path: str, output_dir: str, num_bins: int = 10):
    """
    Process MMLU-Pro CSV: aggregate repeats, compute metrics, generate visualizations.
    
    Accuracy/False Answer Rate: question-level average (correct_count/total_repeats), then mean.
    Brier: per-question average brier across repeats, then mean.
    ECE/Correlation: median of confidences and median of correctness per question.
    """
    raw_data = load_raw_data(input_file_path)
    data = load_and_aggregate_results(input_file_path)
    
    print(f"\n{'='*80}")
    print(f"Processing: {os.path.basename(input_file_path)}")
    print(f"{'='*80}")
    print(f"Total questions: {len(raw_data)}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    api_col = 'api_confidence_key_token'
    api_label = 'API Confidence (key-token)'
    
    # ============================================================================
    # 1. ECE (uses median-aggregated data)
    # ============================================================================
    print(f"\n{'='*80}")
    print("Expected Calibration Error (ECE)")
    print(f"{'='*80}")
    
    if 'self_confidence' in data.columns:
        ece_self = calculate_ece_global(data, 'self_confidence', num_bins=num_bins)
        print(f"\nSelf-Confidence ECE: {ece_self:.6f}")
        print(f"Number of questions: {len(data[['self_confidence', 'correct']].dropna())}")
    
    if api_col in data.columns:
        ece_api = calculate_ece_global(data, api_col, num_bins=num_bins)
        print(f"\n{api_label} ECE: {ece_api:.6f}")
        print(f"Number of questions: {len(data[[api_col, 'correct']].dropna())}")
    
    # ============================================================================
    # 2. Brier Score (from raw: per-question average across repeats, matches calculate_brier_score.py)
    # ============================================================================
    print(f"\n{'='*80}")
    print("Brier Score")
    print(f"{'='*80}")
    
    if 'self_confidences' in raw_data.columns and 'model_answers' in raw_data.columns:
        raw_data['self_brier_score'] = raw_data.apply(
            lambda row: calculate_average_brier_score(
                row['correct_answer'], row['model_answers'], row['self_confidences']
            ),
            axis=1
        )
        brier_scores = raw_data['self_brier_score'].dropna().values
        if len(brier_scores) > 0:
            avg_brier = np.mean(brier_scores)
            std_dev = np.std(brier_scores, ddof=1)
            n = len(brier_scores)
            se = std_dev / np.sqrt(n)
            t_critical = stats.t.ppf(0.975, df=n - 1)
            ci_lower = avg_brier - t_critical * se
            ci_upper = avg_brier + t_critical * se
            ci_margin = t_critical * se
            print(f"\nSelf-Confidence Brier Score:")
            print(f"  Average: {avg_brier:.6f}")
            print(f"  Summary: {avg_brier:.4f}(± {ci_margin:.4f})")
            print(f"  95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
    
    if 'api_confidences_key_token' in raw_data.columns and 'model_answers' in raw_data.columns:
        raw_data['api_brier_score_key_token'] = raw_data.apply(
            lambda row: calculate_average_brier_score(
                row['correct_answer'], row['model_answers'], row['api_confidences_key_token']
            ),
            axis=1
        )
        brier_scores = raw_data['api_brier_score_key_token'].dropna().values
        if len(brier_scores) > 0:
            avg_brier = np.mean(brier_scores)
            std_dev = np.std(brier_scores, ddof=1)
            n = len(brier_scores)
            se = std_dev / np.sqrt(n)
            t_critical = stats.t.ppf(0.975, df=n - 1)
            ci_lower = avg_brier - t_critical * se
            ci_upper = avg_brier + t_critical * se
            ci_margin = t_critical * se
            print(f"\n{api_label} Brier Score:")
            print(f"  Average: {avg_brier:.6f}")
            print(f"  Summary: {avg_brier:.4f}(± {ci_margin:.4f})")
            print(f"  95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
    
    # ============================================================================
    # 3. Task Performance: False Answer Rate (question-level average, then mean)
    # ============================================================================
    print(f"\n{'='*80}")
    print("Task Performance: False Answer Rate")
    print(f"{'='*80}")
    
    if 'model_answers' in raw_data.columns and 'correct_answer' in raw_data.columns:
        raw_data['question_accuracy'] = raw_data.apply(
            lambda row: calculate_question_accuracy(row['correct_answer'], row['model_answers']),
            axis=1
        )
        accuracy_values = raw_data['question_accuracy'].dropna().values
        if len(accuracy_values) > 0:
            avg_accuracy = np.mean(accuracy_values)
            std_dev_accuracy = np.std(accuracy_values, ddof=1)
            n_accuracy = len(accuracy_values)
            se_accuracy = std_dev_accuracy / np.sqrt(n_accuracy)
            t_critical_accuracy = stats.t.ppf(0.975, df=n_accuracy - 1)
            ci_lower_accuracy = avg_accuracy - t_critical_accuracy * se_accuracy
            ci_upper_accuracy = avg_accuracy + t_critical_accuracy * se_accuracy
            ci_margin_accuracy = t_critical_accuracy * se_accuracy
            
            false_answer_rate = 1.0 - avg_accuracy
            ci_lower_far = 1.0 - ci_upper_accuracy
            ci_upper_far = 1.0 - ci_lower_accuracy
            
            print(f"Accuracy: {avg_accuracy:.6f} (question-level avg over repeats, then mean)")
            print(f"False Answer Rate: {false_answer_rate:.6f}")
            print(f"Summary: {false_answer_rate:.4f}(± {ci_margin_accuracy:.4f})")
            print(f"95% CI: [{ci_lower_far:.6f}, {ci_upper_far:.6f}]")
    
    # ============================================================================
    # 4. Correlation
    # ============================================================================
    print(f"\n{'='*80}")
    print("Correlation: Self Confidence vs API Confidence (key-token)")
    print(f"{'='*80}")
    
    pearson_r, pearson_p, pearson_ci, spearman_r, spearman_p, kendall_r, kendall_p = calculate_pearson_coefficient(data, api_col)
    
    if not np.isnan(pearson_r):
        ci_lower, ci_upper = pearson_ci
        margin = (ci_upper - ci_lower) / 2 if not (np.isnan(ci_lower) or np.isnan(ci_upper)) else 0
        corr_str = f"{pearson_r:.4f}(± {margin:.4f})"
    else:
        corr_str = "NaN"
    
    print(f"\n{api_label}: Pearson r = {corr_str}")
    
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate MMLU-Pro experiment results from a CSV file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--result-file", "-f",
        type=str,
        required=True,
        help="Path to the result CSV file to analyze",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Directory for output (default: script_dir/results)",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=10,
        help="Number of bins for ECE (default: 10)",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.output_dir if args.output_dir is not None else os.path.join(script_dir, "outputs")

    input_path = args.result_file
    if not os.path.isabs(input_path) and not os.path.exists(input_path):
        input_path = os.path.join(output_dir, input_path)

    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
    else:
        try:
            process_csv_file(input_path, output_dir, num_bins=args.num_bins)
        except Exception as e:
            print(f"Error processing file: {e}")
            import traceback
            traceback.print_exc()
