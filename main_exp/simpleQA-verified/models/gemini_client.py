"""
Gemini-specific implementation for running experiments on SimpleQA-Verified via OpenRouter.

Mirrors main_exp/TriviaQA/models/gemini_client.py but adapted for:
- Dataset: codelion/SimpleQA-Verified
- Question field: "problem"
- Gold answer field: "answer" (string / list / dict)
- No s_pop / o_pop columns
- Outputs written to main_exp/simpleQA-verified/outputs/
"""

import os
import sys

# Set up path before other imports - add main_exp directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import pandas as pd
import re
from tqdm import tqdm
import ast
import string
import time
import json
from dotenv import load_dotenv
from datetime import datetime

from prompts.prompts import get_experiment_prompt, SYSTEM_PROMPT
from utils.abstain_parser import parse_csv
from utils.response_parser import extract_fields

# Load environment variables
load_dotenv()


class GeminiClientSimpleQAVerified:
    """
    Runner for Gemini experiments via OpenRouter using the OpenAI-compatible API
    on the SimpleQA-Verified dataset.
    """

    # Model configurations
    MODEL_CONFIGS = {
        "gemini-3.1-flash-lite": {
            "full_name": "google/gemini-3.1-flash-lite-preview",
            "batch_size": 50,
            "short_name": "gemini31flashlite"
        }
    }

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    DATASET_SLUG = "simpleqa_verified"  # used in filenames

    def __init__(self, experiments, model_name="gemini-3.1-flash-lite"):
        """
        Initialize Gemini experiment runner.

        Args:
            experiments: Dictionary of experiment configurations
            model_name: Name of the model to use (key in MODEL_CONFIGS)
        """
        self.experiments = experiments
        self.model_name = model_name

        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.MODEL_CONFIGS.keys())}")

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY is not set. Please add it to your .env file or environment."
            )

        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })

        print(f"✅ GeminiClientSimpleQAVerified initialised — model: {self.MODEL_CONFIGS[model_name]['full_name']}")

    # ------------------------
    # HELPER FUNCTIONS
    # ------------------------

    @staticmethod
    def normalize(text):
        return text.lower().translate(str.maketrans('', '', string.punctuation)).strip()

    @staticmethod
    def get_gold_answers(ex):
        """
        Return list of acceptable gold answers.

        SimpleQA-Verified typically stores `answer` as a plain string, but we
        handle list and dict structures defensively.
        """
        ans = ex.get("answer")
        if ans is None:
            return []
        if isinstance(ans, list):
            return [str(x) for x in ans]
        if isinstance(ans, dict):
            values = []
            for k in ["value", "normalized_value", "answer"]:
                if k in ans and ans[k] is not None:
                    values.append(str(ans[k]))
            for v in ans.get("aliases") or []:
                values.append(str(v))
            return values
        return [str(ans)]

    def is_correct(self, llm_answer, gold_answers):
        """Return True if llm_answer matches or is contained in any gold answer."""
        if not gold_answers:
            return False
        if all(isinstance(g, str) and len(g) == 1 for g in gold_answers):
            try:
                joined = "".join(gold_answers)
                parsed = ast.literal_eval(joined)
                gold_answers = [str(parsed)] if not isinstance(parsed, list) else [str(x) for x in parsed]
            except Exception:
                gold_answers = ["".join(gold_answers)]

        if llm_answer is None:
            return False
        norm_llm = self.normalize(llm_answer)
        for gold in gold_answers:
            norm_gold = self.normalize(gold)
            if norm_gold in norm_llm or norm_llm in norm_gold:
                return True
        return False

    # ------------------------
    # GENERATION FUNCTIONS
    # ------------------------

    def generate_response(self, prompt, exp_config, temperature=0, max_tokens=5000,
                          retries=3, retry_delay=5):
        """
        Generate a single response from the model via OpenRouter.

        Args:
            prompt: User prompt string
            exp_config: Experiment config dict (used to decide whether to add system prompt)
            temperature: Sampling temperature (0 = greedy)
            max_tokens: Maximum tokens to generate
            retries: Number of retry attempts on transient errors
            retry_delay: Seconds to wait between retries

        Returns:
            Response text string, or None on failure.
        """
        model_full_name = self.MODEL_CONFIGS[self.model_name]["full_name"]

        messages = []
        if exp_config.get("use_system_prompt", False) and SYSTEM_PROMPT:
            messages.append({"role": "system", "content": SYSTEM_PROMPT})
        messages.append({"role": "user", "content": prompt})

        print("messages:", messages)

        payload = {
            "model": model_full_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        for attempt in range(1, retries + 1):
            try:
                resp = self.session.post(
                    f"{self.OPENROUTER_BASE_URL}/chat/completions",
                    json=payload,
                    timeout=120,
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"⚠️  Attempt {attempt}/{retries} failed: {e}")
                if attempt < retries:
                    time.sleep(retry_delay)
                else:
                    print(f"❌ All {retries} attempts failed. Skipping this example.")
                    return None

    # ------------------------
    # EXPERIMENT FUNCTIONS
    # ------------------------

    def run_batch(self, dataset, start_idx, end_idx, exp_type, reward_correct,
                  reward_abstain, reward_incorrect, model_config):
        """
        Process a slice of the dataset sequentially.

        Args:
            dataset: HuggingFace dataset split
            start_idx: Inclusive start index
            end_idx: Exclusive end index
            exp_type: Experiment type key
            reward_correct: Reward for correct answers
            reward_abstain: Reward for abstaining (used in scheme_b)
            reward_incorrect: Penalty for incorrect answers
            model_config: Model configuration dict

        Returns:
            pd.DataFrame of results
        """
        if exp_type not in self.experiments:
            raise ValueError(f"Unknown experiment type: {exp_type}")

        exp_config = self.experiments[exp_type]
        results = []

        for idx in tqdm(range(start_idx, end_idx), desc=f"Processing [{start_idx}-{end_idx}]"):
            try:
                ex = dataset[idx]
                q = ex["problem"]
                gold_answers = self.get_gold_answers(ex)

                prompt = get_experiment_prompt(
                    reward_correct=reward_correct,
                    reward_abstain=reward_abstain,
                    reward_incorrect=reward_incorrect,
                    question=q,
                    exp_type=exp_type
                )

                out = self.generate_response(prompt, exp_config, temperature=0)

                print("out:", out)

                if not out:
                    print(f"⚠️  No response for index {idx}, skipping.")
                    continue

                ans, conf, best, best_conf = extract_fields(out)
                timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                idk_flag = bool(re.search(r"i don't know", out, re.IGNORECASE))

                if idk_flag:
                    correct = self.is_correct(best, gold_answers)
                else:
                    correct = self.is_correct(ans, gold_answers)

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

                results.append({
                    "dataset_id": idx,
                    "timestamp": timestamp_str,
                    "question": q,
                    "answer": gold_answers,
                    "first_answer": ans, "first_confidence": conf,
                    "best_guess": best, "best_guess_confidence": best_conf,
                    "correct": int(correct), "score": score,
                    "idk_flag": int(idk_flag), "false_answer_flag": false_flag
                })

            except Exception as e:
                print(f"\n⚠️  Error processing sample {idx}: {str(e)}")
                continue

        return pd.DataFrame(results)

    def run_multi_batch_experiment(self, dataset, exp_type, reward_correct, reward_abstain,
                                   reward_incorrect, model_config):
        """
        Run the full experiment in sequential batches and combine results.

        Args:
            dataset: HuggingFace dataset split
            exp_type: Experiment type key
            reward_correct: Reward for correct answers
            reward_abstain: Reward for abstaining
            reward_incorrect: Penalty for incorrect answers
            model_config: Model configuration dict

        Returns:
            Combined pd.DataFrame of all results, or None if no results.
        """
        if exp_type not in self.experiments:
            raise ValueError(f"Unknown experiment type: {exp_type}")

        total_size = len(dataset)
        batch_size = min(total_size, model_config["batch_size"])
        num_batches = (total_size + batch_size - 1) // batch_size

        all_results = []
        tracking_file = (
            f"main_exp/simpleQA-verified/outputs/gemini_batch_files/batch_tracking_{exp_type}.json"
        )
        os.makedirs(os.path.dirname(tracking_file), exist_ok=True)

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_size)

            print(f"\n{'='*80}")
            print(f"Processing batch {i+1}/{num_batches}: indices {start_idx}–{end_idx-1}")
            print(f"{'='*80}")

            results_df = self.run_batch(
                dataset, start_idx, end_idx, exp_type,
                reward_correct, reward_abstain, reward_incorrect, model_config
            )

            if results_df is not None and len(results_df) > 0:
                all_results.append(results_df)
                print(f"✅ Batch {i+1}/{num_batches} done ({len(results_df)} results)")
            else:
                print(f"⚠️  Batch {i+1}/{num_batches} produced no results")

            # Save intermediate tracking info
            tracking_info = {
                "batch": i + 1,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            tracking_data = []
            if os.path.exists(tracking_file):
                with open(tracking_file, "r") as f:
                    tracking_data = json.load(f)
            tracking_data.append(tracking_info)
            with open(tracking_file, "w") as f:
                json.dump(tracking_data, f, indent=2)

            if i < num_batches - 1:
                print(f"\n⏳ Brief pause before next batch…\n")
                time.sleep(2)

        # Combine all results
        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
            combined_df = combined_df.sort_values("dataset_id").reset_index(drop=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if exp_type.lower().startswith("scheme_b"):
                reward_str = f"{reward_correct:+g}_{reward_incorrect:+g}_{reward_abstain:+g}"
            else:
                reward_str = f"{reward_correct:+g}_{reward_incorrect:+g}"

            final_file = (
                f"main_exp/simpleQA-verified/outputs/{model_config['short_name']}_results/"
                f"{self.DATASET_SLUG}_{exp_type}_results_{reward_str}_{timestamp}.csv"
            )
            os.makedirs(os.path.dirname(final_file), exist_ok=True)
            combined_df.to_csv(final_file, index=False)

            # Post-process for scheme_a_baseline and pure_eval
            if exp_type in ["scheme_a_baseline", "pure_eval"]:
                print(f"\n{'='*80}")
                print(f"📝 Post-processing {exp_type} results…")
                print(f"{'='*80}")
                combined_df = parse_csv(final_file, final_file)

            print(f"\n{'='*80}")
            print(f"🎉 All batches processed successfully!")
            print(f"📊 Total results: {len(combined_df)}")
            print(f"💾 Final results saved to: {final_file}")
            print(f"📝 Batch tracking saved to: {tracking_file}")
            print(f"{'='*80}")

            return combined_df
        else:
            print("\n❌ No results to combine")
            return None

    def run_experiment(self, dataset, exp_type, reward_correct, reward_abstain, reward_incorrect):
        """
        Main entry point for Gemini experiments on SimpleQA-Verified.

        Args:
            dataset: Pre-loaded HuggingFace dataset split
            exp_type: Type of experiment to run
            reward_correct: Reward for correct answers
            reward_abstain: Reward for abstaining
            reward_incorrect: Reward/penalty for incorrect answers

        Returns:
            Combined pd.DataFrame of results, or None on failure.
        """
        if self.model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {self.model_name}")

        model_config = self.MODEL_CONFIGS[self.model_name]

        print(f"\n🚀 Running {exp_type.upper()} experiment with {model_config['full_name']} via OpenRouter…")
        print("   Processing sequentially (no batch API)\n")

        try:
            results = self.run_multi_batch_experiment(
                dataset, exp_type, reward_correct, reward_abstain, reward_incorrect,
                model_config
            )
            return results
        except Exception as e:
            print(f"❌ Error during experiment: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
