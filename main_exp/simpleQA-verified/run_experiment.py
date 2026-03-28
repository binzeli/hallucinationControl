"""
Runner for the SimpleQA-Verified dataset (codelion/SimpleQA-Verified).

This mirrors `main_exp/simpleQA/run_experiment.py` but uses a dataset-specific
GPT client so outputs do not collide with the original SimpleQA runs.
"""

import argparse
import sys
import os

# Add main_exp directory to path so imports work correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
import random
import numpy as np

from models.gpt_client_no_batch import GPTClientSimpleQAVerifiedNoBatch
from models.llama_client import LlamaClientSimpleQAVerified
from models.qwen_client import QwenClientSimpleQAVerified
from models.gemini_client import GeminiClientSimpleQAVerified


AVAILABLE_MODELS = {
    "gpt-4o-mini": "gpt",
    "gpt-5-mini": "gpt",
    "llama-3": "llama",
    "qwen-3": "qwen",
    "qwen-3-4b": "qwen",
    "qwen-3.5": "qwen",
    "gemini-3.1-flash-lite": "gemini",
}

EXPERIMENTS = {
    "scheme_a_baseline": {
        "description": "Baseline with no IDK mention",
        "use_system_prompt": False,
    },
    "scheme_a": {
        "description": "No incentive for IDK",
        "use_system_prompt": False,
    },
    "scheme_b": {
        "description": "With IDK incentive",
        "use_system_prompt": False,
    },
    "scheme_b_norm": {
        "description": "B with normative system prompt",
        "use_system_prompt": True,
    },
    "pure_eval": {
        "description": "Pure evaluation with no rewards",
        "use_system_prompt": False,
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Run hallucination control experiments on SimpleQA-Verified.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 main_exp/simpleQA-verified/run_experiment.py --model gpt-4o-mini --scheme scheme_b --samples 10 -rc 1 -ra 0.4 -ri -1
  python3 main_exp/simpleQA-verified/run_experiment.py -m gpt-5-mini -s pure_eval -n 500
  python3 main_exp/simpleQA-verified/run_experiment.py -m gpt-5-mini -s scheme_b -rc 1 -ra 0.4 -ri -1
  python3 main_exp/simpleQA-verified/run_experiment.py -m llama-3 -s scheme_b -rc 1 -ra 0.4 -ri -1
        """,
    )

    parser.add_argument(
        "--model",
        "-m",
        choices=list(AVAILABLE_MODELS.keys()),
        required=True,
        help="Model to use for the experiment",
    )

    parser.add_argument(
        "--scheme",
        "-s",
        choices=list(EXPERIMENTS.keys()),
        default="scheme_b",
        help="Experiment scheme to run (default: scheme_b)",
    )

    parser.add_argument(
        "--samples",
        "-n",
        type=int,
        default=None,
        help="Number of samples to run (default: full dataset)",
    )

    parser.add_argument(
        "--reward-correct",
        "-rc",
        type=float,
        default=0.0,
        help="Reward for correct answer",
    )

    parser.add_argument(
        "--reward-abstain",
        "-ra",
        type=float,
        default=0.4,
        help="Reward for saying 'I don't know'",
    )

    parser.add_argument(
        "--reward-incorrect",
        "-ri",
        type=float,
        default=-1.0,
        help="Reward/penalty for incorrect answer",
    )

    args = parser.parse_args()

    model_type = AVAILABLE_MODELS[args.model]

    print(f"\n{'='*80}")
    print("🚀 Starting Experiment (SimpleQA-Verified)")
    print(f"{'='*80}")
    print(f"Model: {args.model} (type: {model_type})")
    print(f"Scheme: {args.scheme}")
    print(f"Samples: {args.samples if args.samples else 'Full dataset'}")

    reward_str = f"Rewards: Correct={args.reward_correct:+g}, Incorrect={args.reward_incorrect:+g}"
    if args.scheme.lower().startswith("scheme_b"):
        reward_str += f", Abstain={args.reward_abstain:+g}"
    print(reward_str)
    print(f"{'='*80}\n")

    # Load dataset
    print("📚 Loading SimpleQA-Verified dataset...")
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)

    dataset_dict = load_dataset("codelion/SimpleQA-Verified")
    if "test" in dataset_dict:
        split_name = "test"
    elif "train" in dataset_dict:
        split_name = "train"
    else:
        split_name = list(dataset_dict.keys())[0]
    dataset = dataset_dict[split_name]

    if args.samples:
        print(f"⚙️  Running with {args.samples} samples only")
        dataset = dataset.select(range(args.samples))

    print(f"✅ Dataset loaded: {len(dataset)} samples\n")

    if model_type == "gpt":
        runner = GPTClientSimpleQAVerifiedNoBatch(experiments=EXPERIMENTS)
        runner.run_experiment(
            model_name=args.model,
            exp_type=args.scheme,
            n_samples=None,  # dataset already sliced above
            reward_correct=args.reward_correct,
            reward_abstain=args.reward_abstain,
            reward_incorrect=args.reward_incorrect,
        )
    elif model_type == "llama":
        runner = LlamaClientSimpleQAVerified(experiments=EXPERIMENTS, model_name=args.model)
        runner.run_experiment(
            dataset=dataset,
            exp_type=args.scheme,
            reward_correct=args.reward_correct,
            reward_abstain=args.reward_abstain,
            reward_incorrect=args.reward_incorrect,
        )
    elif model_type == "qwen":
        runner = QwenClientSimpleQAVerified(experiments=EXPERIMENTS, model_name=args.model)
    elif model_type == "gemini":
        runner = GeminiClientSimpleQAVerified(experiments=EXPERIMENTS, model_name=args.model)
        runner.run_experiment(
            dataset=dataset,
            exp_type=args.scheme,
            reward_correct=args.reward_correct,
            reward_abstain=args.reward_abstain,
            reward_incorrect=args.reward_incorrect,
        )
    else:
        print(f"❌ Model type '{model_type}' not yet implemented")
        sys.exit(1)


if __name__ == "__main__":
    main()
