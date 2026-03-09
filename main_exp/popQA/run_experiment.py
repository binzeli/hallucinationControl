"""
Main experiment runner that can work with different model types.
This script routes to the appropriate model-specific implementation.
"""

import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gpt_client import GPTClient


AVAILABLE_MODELS = {
    "gpt-4o-mini": "gpt",
    "gpt-5-mini": "gpt",
    # Future models can be added here:
    # "claude-3-5-sonnet": "claude",
    # "gemini-2.0-flash": "gemini",
}

EXPERIMENTS = {
    "scheme_a_baseline": {
        "description": "Baseline with no IDK mention",
        "use_system_prompt": False
    },
    "scheme_a": {
        "description": "No incentive for IDK",
        "use_system_prompt": False
    },
    "scheme_b": {
        "description": "With IDK incentive",
        "use_system_prompt": False
    },
    "scheme_b_norm": {
        "description": "B with normative system prompt",
        "use_system_prompt": True
    },
    "pure_eval": {
        "description": "Pure evaluation with no rewards",
        "use_system_prompt": False
    }
}


def main():
    parser = argparse.ArgumentParser(
        description="Run hallucination control experiments with different models and schemes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiment.py --model gpt-5-mini --scheme scheme_b
  python run_experiment.py --model gpt-4o-mini --scheme scheme_a --samples 1000
  python run_experiment.py -m gpt-5-mini -s pure_eval -n 500
  python run_experiment.py -m gpt-5-mini -s scheme_b -rc 1 -ra 0.4 -ri -1
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        choices=list(AVAILABLE_MODELS.keys()),
        required=True,
        help="Model to use for the experiment"
    )
    
    parser.add_argument(
        "--scheme", "-s",
        choices=list(EXPERIMENTS.keys()),
        default="scheme_b",
        help="Experiment scheme to run (default: scheme_b)"
    )
    
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=None,
        help="Number of samples to run (default: full dataset)"
    )
    
    parser.add_argument(
        "--reward-correct", "-rc",
        type=float,
        default=1.0,
        help="Reward for correct answer (default: 1.0)"
    )
    
    parser.add_argument(
        "--reward-abstain", "-ra",
        type=float,
        default=0.4,
        help="Reward for saying 'I don't know' (default: 0.4)"
    )
    
    parser.add_argument(
        "--reward-incorrect", "-ri",
        type=float,
        default=-1.0,
        help="Reward/penalty for incorrect answer (default: -1.0)"
    )
    
    args = parser.parse_args()
    
    # Get model type
    model_type = AVAILABLE_MODELS[args.model]
    
    print(f"\n{'='*80}")
    print(f"🚀 Starting Experiment")
    print(f"{'='*80}")
    print(f"Model: {args.model} (type: {model_type})")
    print(f"Scheme: {args.scheme}")
    print(f"Samples: {args.samples if args.samples else 'Full dataset'}")
    print(f"Rewards: Correct={args.reward_correct:+g}, Abstain={args.reward_abstain:+g}, Incorrect={args.reward_incorrect:+g}")
    print(f"{'='*80}\n")
    
    # Route to appropriate model handler
    if model_type == "gpt":
        runner = GPTClient(experiments=EXPERIMENTS)
        runner.run_experiment(
            model_name=args.model,
            exp_type=args.scheme,
            n_samples=args.samples,
            reward_correct=args.reward_correct,
            reward_abstain=args.reward_abstain,
            reward_incorrect=args.reward_incorrect
        )
    # Future model types can be added here:
    # elif model_type == "claude":
    #     run_claude_experiment(...)
    # elif model_type == "gemini":
    #     run_gemini_experiment(...)
    else:
        print(f"❌ Model type '{model_type}' not yet implemented")
        sys.exit(1)


if __name__ == "__main__":
    main()
