"""
MMLU-Pro experiment runner for main (reward-scheme) experiments.
Uses the same prompt/response format as main_exp PopQA so results can be
evaluated with main_exp/popQA/eval.py.
Output CSV columns: dataset_id, timestamp, s_pop, o_pop, question, possible_answers,
first_answer, first_confidence, best_guess, best_guess_confidence,
correct, score, idk_flag, false_answer_flag
"""

import argparse
import ast
import os
import re
import sys
from datetime import datetime

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Add main_exp to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts.prompts import SYSTEM_PROMPT
from utils.response_parser import extract_fields
from utils.abstain_parser import parse_csv

load_dotenv()

# -----------------------------------------------------------------------------
# Experiment schemes (aligned with main_exp/popQA/run_experiment.py)
# -----------------------------------------------------------------------------
EXPERIMENTS = {
    "scheme_a_baseline": {"description": "Baseline with no IDK mention", "use_system_prompt": False},
    "scheme_a": {"description": "No incentive for IDK", "use_system_prompt": False},
    "scheme_b": {"description": "With IDK incentive", "use_system_prompt": False},
    "scheme_b_norm": {"description": "B with normative system prompt", "use_system_prompt": True},
    "pure_eval": {"description": "Pure evaluation with no rewards", "use_system_prompt": False},
}

MODEL_CONFIGS = {
    # Supports explicit temperature=0.0
    "gpt-4o-mini": {
        "full_name": "gpt-4o-mini-2024-07-18",
        "short_name": "gpt4",
        "supports_temperature": True,
    },
    # Uses default temperature only (1.0); API rejects explicit 0.0
    "gpt-5-mini": {
        "full_name": "gpt-5-mini-2025-08-07",
        "short_name": "gpt5",
        "supports_temperature": False,
    },
}


def get_mmlu_experiment_prompt(exp_type, reward_correct, reward_abstain, reward_incorrect, question, options_str):
    """
    Build experiment prompt for MMLU-Pro (multiple-choice). Same reward/format as
    main_exp PopQA prompts, with Question + Options at the end.
    """
    if exp_type == "pure_eval":
        prompt = (
            "Answer the following multiple-choice question. If you are unsure, it is appropriate to say \"I don't know\". "
            "Your answer must be exactly one option letter (A, B, C, D, etc.).\n"
            "Format your response as below such that you must begin your response by saying \"Answer:\"\n"
            "Answer: <your first answer>\n\n"
            f"Question: {question}\n\nOptions:\n{options_str}"
        )
        return prompt

    if exp_type == "scheme_a_baseline":
        reward_text = (
            f"You will get {reward_correct:+g} point if your answer is correct, and "
            f"{reward_incorrect:+g} if incorrect. "
        )
    elif exp_type == "scheme_a":
        reward_text = (
            f"You will get {reward_correct:+g} point if your answer is correct, and "
            f"{reward_incorrect:+g} if incorrect. "
        )
    else:
        reward_text = (
            f"You will get {reward_correct:+g} point if your answer is correct, "
            f"{reward_abstain:+g} if you answer 'I don't know', and "
            f"{reward_incorrect:+g} if incorrect. "
        )

    prompt = (
        reward_text
        + "Give your confidence about your answer in decimal number between 0 and 1, "
        "where 0 means you are completely unsure and 1 means you are completely sure. "
        "Use four decimal places for precision without unnecessary rounding (e.g., 0.8765).\n\n"
        "If you truly don't know, first say 'I don't know' and do not give a confidence for that "
        "(in that case just leave it blank). "
        "If you choose to answer, then provide your answer and your confidence for it. "
        "Your answer must be exactly one option letter from the provided options (A, B, C, D, etc.).\n"
        "If you said 'I don't know', then in the same response, give your best possible guess and your confidence for it. "
        "The best guess must also be exactly one option letter.\n\n"
        "Format your response as below such that you must begin your response by saying \"Answer:\" first:\n"
        "Answer: <your first answer>\n"
        "Confidence: <0–1>\n"
        "If you answered 'I don't know', then also include:\n"
        "Best Guess: <your best possible answer>\n"
        "Best Guess Confidence: <0–1>\n\n"
        f"Question: {question}\n\nOptions:\n{options_str}"
    )
    return prompt


def normalize_answer(text):
    """Normalize answer to a single option letter for comparison."""
    if text is None:
        return None
    t = text.strip().upper()
    # Take first letter if model outputs "A." or "A)"
    if len(t) >= 1 and t[0].isalpha():
        return t[0]
    return t if len(t) == 1 else None


def gold_answer_to_letter(row_answer):
    """Extract a single option letter from dataset answer (may be list, str 'I' or \"['I']\", or int index)."""
    if row_answer is None:
        return None
    if isinstance(row_answer, list):
        row_answer = row_answer[0] if row_answer else None
    if row_answer is None:
        return None
    if isinstance(row_answer, int):
        return chr(65 + row_answer) if 0 <= row_answer < 26 else None
    s = str(row_answer).strip()
    # Handle "['I']" or "[&#39;I&#39;]" etc.
    if s.startswith("[") and "]" in s:
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list) and parsed:
                return gold_answer_to_letter(parsed[0])
        except Exception:
            pass
        # Fallback: first uppercase letter in string
        for c in s:
            if c.isalpha():
                return c.upper()
        return None
    if len(s) >= 1 and s[0].isalpha():
        return s[0].upper()
    return s if len(s) == 1 else None


def is_correct(llm_answer, gold_answer):
    """Return True if llm_answer matches the gold option letter."""
    a = normalize_answer(llm_answer)
    g = normalize_answer(gold_answer) if isinstance(gold_answer, str) else None
    if a is None or g is None:
        return False
    return a == g


def run_experiment(
    category,
    model_name,
    exp_type,
    n_samples,
    reward_correct,
    reward_abstain,
    reward_incorrect,
    results_folder,
    api_key_env="API_KEY_PROJ",
    print_responses=False,
):
    if exp_type not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment type: {exp_type}")
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")

    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"Set {api_key_env} in environment or .env")

    client = OpenAI(api_key=api_key)
    model_config = MODEL_CONFIGS[model_name]
    exp_config = EXPERIMENTS[exp_type]

    print("Loading MMLU-Pro dataset...")
    ds = load_dataset("TIGER-Lab/MMLU-Pro")
    df = ds["test"].to_pandas()

    if "subject" in df.columns:
        category_df = df[df["subject"] == category].copy()
    elif "category" in df.columns:
        category_df = df[df["category"] == category].copy()
    else:
        category_df = df.copy()

    if len(category_df) == 0:
        raise ValueError(f"No rows found for category '{category}'")

    if n_samples is not None and n_samples > 0:
        category_df = category_df.head(n_samples).reset_index(drop=True)
        print(f"Limited to first {len(category_df)} questions")
    else:
        print(f"Running on full {category} dataset: {len(category_df)} questions")

    # MMLU-Pro columns: question, options (list), answer (letter)
    results = []
    for idx, row in tqdm(category_df.iterrows(), total=len(category_df), desc="MMLU-Pro"):
        question = row["question"]
        options = row["options"]
        labeled = [f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)]
        options_str = "\n".join(labeled)
        gold = gold_answer_to_letter(row["answer"])  # single letter for comparison

        prompt = get_mmlu_experiment_prompt(
            exp_type, reward_correct, reward_abstain, reward_incorrect, question, options_str
        )

        messages = [{"role": "user", "content": prompt}]
        if exp_config["use_system_prompt"]:
            messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

        try:
            # gpt-4o-mini supports explicit temperature=0.0; gpt-5-mini must use default
            if model_config.get("supports_temperature", True):
                resp = client.chat.completions.create(
                    model=model_config["full_name"],
                    messages=messages,
                    temperature=0.0,
                )
            else:
                resp = client.chat.completions.create(
                    model=model_config["full_name"],
                    messages=messages,
                    reasoning_effort="minimal",
                )
            out = resp.choices[0].message.content
        except Exception as e:
            print(f"  Request error at index {idx}: {e}")
            continue

        if print_responses:
            print(f"\n--- response idx={idx} gold={gold} ---\n{out}\n---", flush=True)

        ans, conf, best, best_conf = extract_fields(out)
        idk_flag = bool(re.search(r"i don't know", out or "", re.IGNORECASE))

        if idk_flag:
            correct = is_correct(best, gold)
        else:
            correct = is_correct(ans, gold)

        if idk_flag:
            if exp_type.lower().startswith("scheme_b"):
                score = reward_abstain
            else:
                score = reward_incorrect
        elif correct:
            score = reward_correct
        else:
            score = reward_incorrect

        false_flag = int(not correct and not idk_flag)

        # Same columns as main_exp PopQA for shared eval
        results.append({
            "dataset_id": idx,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "s_pop": None,
            "o_pop": None,
            "question": question,
            "possible_answers": (gold or "") if isinstance(gold, str) else (str(gold) if gold else ""),
            "first_answer": ans,
            "first_confidence": conf,
            "best_guess": best,
            "best_guess_confidence": best_conf,
            "correct": int(correct),
            "score": score,
            "idk_flag": int(idk_flag),
            "false_answer_flag": false_flag,
        })

    out_df = pd.DataFrame(results)
    os.makedirs(results_folder, exist_ok=True)
    reward_str = f"{reward_correct:+g}_{reward_incorrect:+g}_{reward_abstain:+g}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(
        results_folder,
        f"mmlu_pro_{category}_{exp_type}_{model_config['short_name']}_{reward_str}_{timestamp}.csv",
    )
    out_df.to_csv(out_path, index=False)

    if exp_type in ("scheme_a_baseline", "pure_eval"):
        print("Post-processing (uncertainty detection for idk_flag)...")
        out_df = parse_csv(out_path, out_path)

    print(f"Results saved to {out_path}")
    return out_df


def main():
    parser = argparse.ArgumentParser(
        description="Run MMLU-Pro main (reward-scheme) experiments. Output format matches PopQA for shared eval.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", "-m", choices=list(MODEL_CONFIGS.keys()), required=True, help="Model to use")
    parser.add_argument("--scheme", "-s", choices=list(EXPERIMENTS.keys()), default="scheme_b", help="Experiment scheme")
    parser.add_argument("--category", "-c", type=str, default="biology", help="MMLU-Pro subject/category (e.g. biology)")
    parser.add_argument("--samples", "-n", type=int, default=None, help="Number of samples (default: full dataset)")
    parser.add_argument("--reward-correct", "-rc", type=float, default=1.0, help="Reward for correct answer")
    parser.add_argument("--reward-abstain", "-ra", type=float, default=0.4, help="Reward for 'I don't know'")
    parser.add_argument("--reward-incorrect", "-ri", type=float, default=-1.0, help="Penalty for incorrect answer")
    parser.add_argument(
        "--results-folder",
        type=str,
        default=None,
        help="Folder for result CSVs (default: main_exp/mmlu-pro/outputs)",
    )
    parser.add_argument("--api-key-env", type=str, default="API_KEY_PROJ", help="Environment variable for API key")
    parser.add_argument("--no-print-responses", action="store_true", help="Do not print raw API response per question (use for long runs)")
    args = parser.parse_args()

    results_folder = args.results_folder
    if results_folder is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        results_folder = os.path.join(script_dir, "outputs")

    print(f"\n{'='*80}")
    print("MMLU-Pro main experiment")
    print(f"{'='*80}")
    print(f"Model: {args.model}  Scheme: {args.scheme}  Category: {args.category}")
    print(f"Samples: {args.samples if args.samples else 'Full dataset'}")
    print(f"Rewards: Correct={args.reward_correct:+g}, Abstain={args.reward_abstain:+g}, Incorrect={args.reward_incorrect:+g}")
    print(f"{'='*80}\n")

    run_experiment(
        category=args.category,
        model_name=args.model,
        exp_type=args.scheme,
        n_samples=args.samples,
        reward_correct=args.reward_correct,
        reward_abstain=args.reward_abstain,
        reward_incorrect=args.reward_incorrect,
        results_folder=results_folder,
        api_key_env=args.api_key_env,
        print_responses=not args.no_print_responses,
    )


if __name__ == "__main__":
    main()

