"""
LLaMA 3-specific implementation for running experiments with Hugging Face models.
"""

import os
import sys

# Set up path before other imports - add main_exp directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import re
from tqdm import tqdm
from datasets import load_dataset
import ast
import string
import random
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
import json
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login

from prompts.prompts import get_experiment_prompt, SYSTEM_PROMPT
from utils.abstain_parser import parse_csv
from utils.response_parser import extract_fields

# Load environment variables
load_dotenv()


class LlamaClient:
    """
    Runner for LLaMA 3 experiments using Hugging Face models.
    """

    # Model configurations
    MODEL_CONFIGS = {
        "llama-3": {
            "full_name": "meta-llama/Meta-Llama-3-8B-Instruct",
            "batch_size": 40,
            "short_name": "llama3-8b"
        }
    }

    def __init__(self, experiments, model_name="llama-3"):
        """
        Initialize LLaMA experiment runner.

        Args:
            experiments: Dictionary of experiment configurations
            model_name: Name of the model to use
        """
        self.experiments = experiments
        self.model_name = model_name

        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}")

        model_config = self.MODEL_CONFIGS[model_name]
        self.model_full_name = model_config["full_name"]

        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            login(token=hf_token)

        print(f"🔧 Loading model: {self.model_full_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_full_name, token=hf_token)
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_full_name,
            torch_dtype=torch.float16,
            device_map="auto",
            token=hf_token
        )
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        print(f"✅ Model loaded successfully")

    # ------------------------
    # HELPER FUNCTIONS
    # ------------------------

    @staticmethod
    def normalize(text):
        return text.lower().translate(str.maketrans('', '', string.punctuation)).strip()

    def is_correct(self, llm_answer, gold_answers):
        """Return True if llm_answer matches or is contained in any gold answer."""
        if all(isinstance(g, str) and len(g) == 1 for g in gold_answers):
            try:
                joined = "".join(gold_answers)
                gold_answers = ast.literal_eval(joined)
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

    def generate_response(self, prompt, exp_type=None, temperature=0, max_tokens=500):
        """Generate a single response from the model."""
        use_system_prompt = self.experiments.get(exp_type, {}).get("use_system_prompt", False)
        messages = []
        if use_system_prompt and SYSTEM_PROMPT:
            messages.append({"role": "system", "content": SYSTEM_PROMPT})
        messages.append({"role": "user", "content": prompt})

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        do_sample = temperature is not None and temperature > 0
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs["temperature"] = float(temperature)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs,
                **gen_kwargs
            )

        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response

    def generate_responses_batch(self, prompts, exp_type=None, temperature=0, max_tokens=500):
        """Generate responses for a batch of prompts in one call."""
        use_system_prompt = self.experiments.get(exp_type, {}).get("use_system_prompt", False)
        messages_list = []
        for prompt in prompts:
            messages = []
            if use_system_prompt and SYSTEM_PROMPT:
                messages.append({"role": "system", "content": SYSTEM_PROMPT})
            messages.append({"role": "user", "content": prompt})
            messages_list.append(messages)
        texts = [
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            for messages in messages_list
        ]

        print("messages_list:", messages_list[:2])  # Debug: print first 2 message lists
        model_inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True
        ).to(self.model.device)

        do_sample = temperature is not None and temperature > 0
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs["temperature"] = float(temperature)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs,
                **gen_kwargs
            )

        input_lengths = (model_inputs.input_ids != self.tokenizer.pad_token_id).sum(dim=1).tolist()
        trimmed = [
            output_ids[input_len:]
            for output_ids, input_len in zip(generated_ids, input_lengths)
        ]
        responses = self.tokenizer.batch_decode(trimmed, skip_special_tokens=True)

        print("Generated responses:", responses[:2])  # Debug: print first 2 responses
        return responses

    def run_batch(self, dataset, start_idx, end_idx, exp_type, reward_correct, reward_abstain,
                  reward_incorrect, model_config):
        """Process a batch of examples."""
        if exp_type not in self.experiments:
            raise ValueError(f"Unknown experiment type: {exp_type}")

        results = []

        indices = list(range(start_idx, end_idx))
        print(f"📊 Total indices to process: {len(indices)}")

        for i in tqdm(range(0, len(indices), model_config["batch_size"]), desc=f"Processing batch [{start_idx}-{end_idx}]"):
            batch_indices = indices[i:i + model_config["batch_size"]]
            print(f"\n   🔄 Sub-batch {i // model_config['batch_size'] + 1}: Processing {len(batch_indices)} examples (indices {batch_indices[0]}-{batch_indices[-1]})")

            prompts = []
            examples = []
            for idx in batch_indices:
                ex = dataset[idx]
                q = ex["question"]
                gold = ex["possible_answers"]
                s_pop = ex.get("s_pop", None)
                o_pop = ex.get("o_pop", None)
                prompt = get_experiment_prompt(
                    reward_correct=reward_correct,
                    reward_abstain=reward_abstain,
                    reward_incorrect=reward_incorrect,
                    question=q,
                    exp_type=exp_type
                )
                prompts.append(prompt)
                examples.append((idx, q, gold, s_pop, o_pop))

            try:
                print(f"      🤖 Generating responses for {len(prompts)} examples...")
                outputs = self.generate_responses_batch(prompts, exp_type=exp_type, temperature=0)
                print(f"      ✅ Responses generated successfully")
            except Exception as e:
                print(f"⚠️  Error processing batch {i // model_config['batch_size']}: {str(e)}")
                continue

            print(f"      📝 Processing results...")
            processed_count = 0
            for (idx, q, gold, s_pop, o_pop), out, prompt in zip(examples, outputs, prompts):
                try:
                    print(f"         [{processed_count + 1}/{len(examples)}] Processing example {idx}...")
                    print("         prompt:", prompt[:100] + "..." if len(prompt) > 100 else prompt)
                    print("         response:", out[:100] + "..." if len(out) > 100 else out)

                    if not out:
                        print(f"⚠️  Could not generate response for index {idx}")
                        continue

                    ans, conf, best, best_conf = extract_fields(out)
                    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    idk_flag = bool(re.search(r"i don't know", out or "", re.IGNORECASE))

                    if idk_flag:
                        correct = self.is_correct(best, gold)
                    else:
                        correct = self.is_correct(ans, gold)

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
                        "s_pop": s_pop, "o_pop": o_pop,
                        "question": q, "possible_answers": gold,
                        "first_answer": ans, "first_confidence": conf,
                        "best_guess": best, "best_guess_confidence": best_conf,
                        "correct": int(correct), "score": score,
                        "idk_flag": int(idk_flag), "false_answer_flag": false_flag
                    })
                    processed_count += 1
                    print(f"         ✅ Example {idx} processed (correct: {correct}, idk: {idk_flag})")
                except Exception as e:
                    print(f"⚠️  Error processing index {idx}: {str(e)}")
                    continue

            print(f"      🎯 Sub-batch complete: {processed_count}/{len(examples)} examples processed\n")

        return pd.DataFrame(results)

    def run_multi_batch_experiment(self, dataset, exp_type, reward_correct, reward_abstain, reward_incorrect,
                                   model_config):
        """Run experiment in multiple batches sequentially."""
        if exp_type not in self.experiments:
            raise ValueError(f"Unknown experiment type: {exp_type}")

        total_size = len(dataset)
        batch_size = min(total_size, model_config["batch_size"])
        num_batches = (total_size + batch_size - 1) // batch_size

        all_results = []
        tracking_file = f"main_exp/popQA/outputs/llama_batch_files/batch_tracking_{exp_type}.json"

        os.makedirs(os.path.dirname(tracking_file), exist_ok=True)

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_size)

            print(f"\n{'='*80}")
            print(f"Processing batch {i+1}/{num_batches}: {start_idx} to {end_idx}")
            print(f"{'='*80}")

            results_df = self.run_batch(dataset, start_idx, end_idx, exp_type,
                                        reward_correct, reward_abstain, reward_incorrect, model_config)

            if results_df is not None and len(results_df) > 0:
                all_results.append(results_df)
                print(f"✅ Batch {i+1}/{num_batches} processed successfully ({len(results_df)} results)")
            else:
                print(f"⚠️  Batch {i+1}/{num_batches} produced no results")

            if i < num_batches - 1:
                print(f"\n⏳ Brief pause before next batch...\n")
                time.sleep(2)

        # Combine all results
        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
            combined_df = combined_df.sort_values('dataset_id').reset_index(drop=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Format rewards for filename
            if exp_type.lower().startswith("scheme_b"):
                reward_str = f"{reward_correct:+g}_{reward_incorrect:+g}_{reward_abstain:+g}"
            else:
                reward_str = f"{reward_correct:+g}_{reward_incorrect:+g}"
            final_file = f"main_exp/popQA/outputs/{model_config['short_name']}_results/popqa_{exp_type}_results_{reward_str}_{timestamp}.csv"

            os.makedirs(os.path.dirname(final_file), exist_ok=True)
            combined_df.to_csv(final_file, index=False)

            # Post-process for scheme_a_baseline and pure_eval
            if exp_type in ["scheme_a_baseline", "pure_eval"]:
                print(f"\n{'='*80}")
                print(f"📝 Post-processing {exp_type} results...")
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
        """Main entry point for LLaMA experiments.

        Args:
            dataset: Pre-loaded dataset to run experiment on
            exp_type: Type of experiment to run
            reward_correct: Reward for correct answers
            reward_abstain: Reward for abstaining
            reward_incorrect: Reward/penalty for incorrect answers
        """

        # Get model configuration
        if self.model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {self.model_name}")

        model_config = self.MODEL_CONFIGS[self.model_name]

        print(f"\n🚀 Running {exp_type.upper()} experiment with LLaMA 3...")
        print("   Processing batches sequentially\n")

        try:
            results = self.run_multi_batch_experiment(
                dataset, exp_type, reward_correct, reward_abstain, reward_incorrect,
                model_config
            )
            return results
        except Exception as e:
            print(f"❌ Error during batch processing: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
