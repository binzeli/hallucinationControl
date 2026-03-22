"""
GPT-specific implementation for running experiments on SimpleQA-Verified (non-batch).

This mirrors `main_exp/TriviaQA/models/gpt_client_no_batch.py`:
- calls the LLM one at a time (no OpenAI Batch API)
- parses the same response format (Answer/Confidence/Best Guess + confidences)
- writes outputs to `main_exp/simpleQA-verified/outputs/...`
"""

import os
import sys

# Set up path before other imports - add main_exp directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
import pandas as pd
import re
from tqdm import tqdm
import string
import random
import numpy as np
from dotenv import load_dotenv
from datetime import datetime

from datasets import load_dataset

from prompts.prompts import get_experiment_prompt, SYSTEM_PROMPT
from utils.response_parser import extract_fields
from utils.abstain_parser import parse_csv

# Load environment variables
load_dotenv()


class GPTClientSimpleQAVerifiedNoBatch:
    """
    Runner for GPT experiments using OpenAI models on SimpleQA-Verified.
    Non-batch version: calls the LLM sequentially for each sample.
    """

    MODEL_CONFIGS = {
        "gpt-4o-mini": {
            "full_name": "gpt-4o-mini-2024-07-18",
            "short_name": "gpt4",
        },
        "gpt-5-mini": {
            "full_name": "gpt-5-mini-2025-08-07",
            "short_name": "gpt5",
        },
    }

    DATASET_ID = "codelion/SimpleQA-Verified"
    DATASET_SLUG = "simpleqa_verified"  # used in filenames

    def __init__(self, experiments):
        self.experiments = experiments
        self.client = OpenAI(api_key=os.getenv("API_KEY"))

    # ------------------------
    # HELPER FUNCTIONS
    # ------------------------

    @staticmethod
    def normalize(text):
        return text.lower().translate(str.maketrans("", "", string.punctuation)).strip()

    @staticmethod
    def get_gold_answers(ex):
        """
        Return list of acceptable gold answers.

        SimpleQA-Verified typically has `answer` as a string, but we allow
        list/dict-like structures defensively.
        """
        ans = ex.get("answer")
        if ans is None:
            return []
        if isinstance(ans, list):
            return [str(x) for x in ans]
        if isinstance(ans, dict):
            # Try common keys if present
            values = []
            for k in ["value", "normalized_value", "answer"]:
                if k in ans and ans[k] is not None:
                    values.append(ans[k])
            for v in ans.get("aliases") or []:
                values.append(v)
            return [str(v) for v in values if v is not None]
        return [str(ans)]

    def is_correct(self, llm_answer, gold_answers):
        """Return True if llm_answer matches or is contained in any gold answer."""
        if not gold_answers:
            return False

        # If gold answers are stored as a serialized list, expand them.
        if all(isinstance(g, str) and len(g) == 1 for g in gold_answers):
            try:
                joined = "".join(gold_answers)
                import ast

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
    # LLM CALL FUNCTIONS
    # ------------------------

    def call_llm(self, prompt, model_name, model_config, system_prompt=None):
        """
        Call the model for a single prompt.

        Uses responses API for GPT-5 and chat completions for GPT-4.
        """
        if "gpt-5" in model_name:
            input_messages = []
            if system_prompt:
                input_messages.append({"role": "system", "content": system_prompt})
            input_messages.append({"role": "user", "content": prompt})

            response = self.client.responses.create(
                model=model_config["full_name"],
                input=input_messages,
                reasoning={"effort": "minimal"},
            )
            return response.output_text

        # GPT-4
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=model_config["full_name"],
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message.content

    # ------------------------
    # EXPERIMENT RUN
    # ------------------------

    def run_experiment_no_batch(self, dataset, exp_type, reward_correct, reward_abstain, reward_incorrect, model_name, model_config):
        if exp_type not in self.experiments:
            raise ValueError(f"Unknown experiment type: {exp_type}")

        exp_config = self.experiments[exp_type]
        system_prompt = SYSTEM_PROMPT if exp_config.get("use_system_prompt") else None

        results = []
        total_samples = len(dataset)

        print(f"\n{'='*80}")
        print(f"Running {exp_type.upper()} experiment - Sequential LLM calls (SimpleQA-Verified)")
        print(f"Total samples: {total_samples}")
        print(f"{'='*80}\n")

        os.makedirs(f"main_exp/simpleQA-verified/outputs/{model_config['short_name']}_results", exist_ok=True)

        for idx in tqdm(range(total_samples), desc="Processing samples"):
            try:
                ex = dataset[idx]
                q = ex["problem"]
                gold_answers = self.get_gold_answers(ex)

                prompt = get_experiment_prompt(
                    reward_correct=reward_correct,
                    reward_abstain=reward_abstain,
                    reward_incorrect=reward_incorrect,
                    question=q,
                    exp_type=exp_type,
                )

                response_text = self.call_llm(prompt, model_name, model_config, system_prompt)
                ans, conf, best, best_conf = extract_fields(response_text)

                timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                idk_flag = bool(re.search(r"i don't know", response_text or "", re.IGNORECASE))

                if idk_flag:
                    correct = self.is_correct(best, gold_answers)
                else:
                    correct = self.is_correct(ans, gold_answers)

                if idk_flag:
                    score = reward_abstain if exp_type.lower().startswith("scheme_b") else reward_incorrect
                elif correct:
                    score = reward_correct
                else:
                    score = reward_incorrect

                false_flag = int((not correct) and (not idk_flag))

                results.append(
                    {
                        "dataset_id": idx,
                        "timestamp": timestamp_str,
                        "question": q,
                        "answer": gold_answers,
                        "first_answer": ans,
                        "first_confidence": conf,
                        "best_guess": best,
                        "best_guess_confidence": best_conf,
                        "correct": int(correct),
                        "score": score,
                        "idk_flag": int(idk_flag),
                        "false_answer_flag": false_flag,
                    }
                )
            except Exception as e:
                print(f"\n⚠️  Error processing sample {idx}: {str(e)}")
                continue

        if not results:
            print("❌ No results collected!")
            return None

        results_df = pd.DataFrame(results)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if exp_type.lower().startswith("scheme_b"):
            reward_str = f"{reward_correct:+g}_{reward_incorrect:+g}_{reward_abstain:+g}"
        else:
            reward_str = f"{reward_correct:+g}_{reward_incorrect:+g}"

        output_file = (
            "main_exp/simpleQA-verified/outputs/"
            f"{model_config['short_name']}_results/"
            f"{self.DATASET_SLUG}_{exp_type}_results_{reward_str}_{timestamp}.csv"
        )
        results_df.to_csv(output_file, index=False)

        # Post-process for scheme_a_baseline and pure_eval
        if exp_type in ["scheme_a_baseline", "pure_eval"]:
            print(f"\n{'='*80}")
            print(f"📝 Post-processing {exp_type} results...")
            print(f"{'='*80}")
            results_df = parse_csv(output_file, output_file)

        print(f"\n{'='*80}")
        print("🎉 Experiment completed!")
        print(f"📊 Total samples processed: {len(results_df)}")
        print(f"Accuracy: {results_df['correct'].mean():.2%}")
        print(f"IDK rate: {results_df['idk_flag'].mean():.2%}")
        print(f"False answer rate: {results_df['false_answer_flag'].mean():.2%}")
        print(f"Average score: {results_df['score'].mean():.4f}")
        print(f"💾 Results saved to: {output_file}")
        print(f"{'='*80}\n")

        return results_df

    def run_experiment(self, model_name, exp_type, n_samples, reward_correct, reward_abstain, reward_incorrect):
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}")

        model_config = self.MODEL_CONFIGS[model_name]

        SEED = 42
        random.seed(SEED)
        np.random.seed(SEED)

        print("📚 Loading SimpleQA-Verified dataset...")
        dataset_dict = load_dataset(self.DATASET_ID)
        # Prefer `test` split for evaluation consistency.
        if "test" in dataset_dict:
            split_name = "test"
        elif "train" in dataset_dict:
            split_name = "train"
        else:
            split_name = list(dataset_dict.keys())[0]
        dataset = dataset_dict[split_name]

        if n_samples:
            print(f"⚙️  Running with {n_samples} samples only")
            dataset = dataset.select(range(n_samples))

        try:
            return self.run_experiment_no_batch(
                dataset=dataset,
                exp_type=exp_type,
                reward_correct=reward_correct,
                reward_abstain=reward_abstain,
                reward_incorrect=reward_incorrect,
                model_name=model_name,
                model_config=model_config,
            )
        except Exception as e:
            print(f"❌ Error during experiment: {str(e)}")
            import traceback

            traceback.print_exc()
            raise

