"""
GPT-specific implementation for running experiments with OpenAI models on GSM8K dataset.
Non-batch version: calls LLM one at a time for each question.
"""

import os
import sys

# Set up path before other imports - add main_exp directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
import pandas as pd
import re
from tqdm import tqdm
from datasets import load_dataset
import string
import random
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
import json
import time

from prompts.prompts import get_experiment_prompt, SYSTEM_PROMPT
from utils.response_parser import extract_fields_gsm8k

# Load environment variables
load_dotenv()


class GPTClient:
    """
    Runner for GPT experiments with OpenAI models on GSM8K dataset.
    Non-batch version: processes one question at a time.
    """
    
    # Model configurations
    MODEL_CONFIGS = {
        "gpt-4o-mini": {
            "full_name": "gpt-4o-mini-2024-07-18",
            "short_name": "gpt4"
        },
        "gpt-5-mini": {
            "full_name": "gpt-5-mini-2025-08-07",
            "short_name": "gpt5"
        }
    }
    
    def __init__(self, experiments):
        """
        Initialize GPT experiment runner.
        
        Args:
            experiments: Dictionary of experiment configurations
        """
        self.experiments = experiments
        self.client = OpenAI(api_key=os.getenv("API_KEY"))
    
    # ------------------------
    # HELPER FUNCTIONS
    # ------------------------
    
    @staticmethod
    def extract_numeric_answer_from_gsm8k(answer_text):
        """
        Extract the numeric answer from GSM8K answer text.
        GSM8K answers have format: ... #### ANSWER
        We extract just the numeric value after ####
        
        Args:
            answer_text: Text containing the answer after "####"
            
        Returns:
            The numeric value as a string
        """
        if not answer_text:
            return None
        
        # Try to extract number after ####
        match = re.search(r'####\s*(.+?)(?:\n|$)', answer_text, re.IGNORECASE)
        if match:
            numeric_part = match.group(1).strip()
            return numeric_part
        
        return None
    
    def is_correct_gsm8k(self, llm_answer, gold_answer):
        """
        Check if LLM answer matches GSM8K gold answer (numeric comparison).
        Both should be numeric values extracted from "####" lines.
        Handles currency symbols, commas, spaces, and units (e.g., "3 bolts" vs "3").
        
        Args:
            llm_answer: LLM's extracted numeric answer
            gold_answer: Gold standard numeric answer
            
        Returns:
            bool: True if answers match
        """
        if llm_answer is None or gold_answer is None:
            return False
        
        # Normalize both answers - remove currency symbols, spaces, commas, etc.
        llm_norm = str(llm_answer).strip().replace('$', '').replace(',', '').replace(' ', '')
        gold_norm = str(gold_answer).strip().replace('$', '').replace(',', '').replace(' ', '')
        
        # Try to extract just the numeric part using regex (handles cases like "3bolts" or "3 bolts")
        llm_numeric = re.search(r'[-+]?[\d.]+', llm_norm)
        gold_numeric = re.search(r'[-+]?[\d.]+', gold_norm)
        
        if llm_numeric and gold_numeric:
            try:
                llm_num = float(llm_numeric.group())
                gold_num = float(gold_numeric.group())
                return abs(llm_num - gold_num) < 1e-6
            except (ValueError, TypeError):
                pass
        
        # Fall back to full string comparison
        return llm_norm.lower() == gold_norm.lower()

    # ------------------------
    # LLM CALL FUNCTIONS
    # ------------------------

    def call_llm(self, prompt, model_name, model_config, system_prompt=None):
        """
        Call LLM with a single prompt.
        Use responses API for GPT-5 models, chat completions API for GPT-4 models.
        
        Args:
            prompt: The user prompt
            model_name: Name of the model to use
            model_config: Model configuration dict
            system_prompt: Optional system prompt
            
        Returns:
            The LLM response text
        """
        # Use responses API for GPT-5, chat completions for GPT-4
        if "gpt-5" in model_name:
            # GPT-5 uses responses API
            input_messages = []
            
            if system_prompt:
                input_messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            input_messages.append({
                "role": "user",
                "content": prompt
            })
            
            response = self.client.responses.create(
                model=model_config["full_name"],
                input=input_messages,
                reasoning={"effort": "minimal"}
            )
            
            return response.output_text
        else:
            # GPT-4 uses chat completions API
            messages = []
            
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            response = self.client.chat.completions.create(
                model=model_config["full_name"],
                messages=messages,
                temperature=0
            )
            
            return response.choices[0].message.content

    def run_experiment_no_batch(self, dataset, exp_type, reward_correct, reward_abstain, 
                               reward_incorrect, model_name, model_config):
        """Run experiment by calling LLM one at a time for each sample."""
        if exp_type not in self.experiments:
            raise ValueError(f"Unknown experiment type: {exp_type}")
        
        exp_config = self.experiments[exp_type]
        system_prompt = SYSTEM_PROMPT if exp_config.get("use_system_prompt") else None
        
        results = []
        total_samples = len(dataset)
        
        print(f"\n{'='*80}")
        print(f"Running {exp_type.upper()} experiment - Sequential LLM calls")
        print(f"Total samples: {total_samples}")
        print(f"{'='*80}\n")
        
        # Create output directory
        os.makedirs(f"main_exp/gms8k/outputs/{model_config['short_name']}_results", exist_ok=True)
        
        for idx in tqdm(range(total_samples), desc="Processing samples"):
            try:
                ex = dataset[idx]
                q = ex["question"]
                gold_answer_text = ex["answer"]
                gold_answer = self.extract_numeric_answer_from_gsm8k(gold_answer_text)
                
                # Get the prompt
                prompt = get_experiment_prompt(
                    reward_correct=reward_correct,
                    reward_abstain=reward_abstain,
                    reward_incorrect=reward_incorrect,
                    question=q,
                    exp_type=exp_type
                )
                
                print("prompt: ", prompt)
                print("system_prompt: ", system_prompt)
                # Call LLM
                response_text = self.call_llm(prompt, model_name, model_config, system_prompt)
                
                print("response_text: ", response_text)
                # Parse response
                ans, conf, best, best_conf = extract_fields_gsm8k(response_text)
                timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                idk_flag = bool(re.search(r"i don't know", response_text or "", re.IGNORECASE))
                
                # Determine correctness
                if idk_flag:
                    llm_answer = best
                    correct = self.is_correct_gsm8k(llm_answer, gold_answer)
                else:
                    llm_answer = ans
                    correct = self.is_correct_gsm8k(llm_answer, gold_answer)
                
                # Calculate score
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
                    "gold_answer": gold_answer,
                    "first_answer": ans,
                    "first_confidence": conf,
                    "best_guess": best,
                    "best_guess_confidence": best_conf,
                    "correct": int(correct),
                    "score": score,
                    "idk_flag": int(idk_flag),
                    "false_answer_flag": false_flag
                })
                
            except Exception as e:
                print(f"\n⚠️  Error processing sample {idx}: {str(e)}")
                continue
        
        # Convert to dataframe
        if not results:
            print("❌ No results collected!")
            return None
        
        results_df = pd.DataFrame(results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if exp_type.lower().startswith("scheme_b"):
            reward_str = f"{reward_correct:+g}_{reward_incorrect:+g}_{reward_abstain:+g}"
        else:
            reward_str = f"{reward_correct:+g}_{reward_incorrect:+g}"
        
        output_file = f"main_exp/gms8k/outputs/{model_config['short_name']}_results/gsm8k_{exp_type}_results_{reward_str}_{timestamp}.csv"
        results_df.to_csv(output_file, index=False)
        
        print(f"\n{'='*80}")
        print(f"🎉 Experiment completed!")
        print(f"📊 Total samples processed: {len(results_df)}")
        print(f"Accuracy: {results_df['correct'].mean():.2%}")
        print(f"IDK rate: {results_df['idk_flag'].mean():.2%}")
        print(f"False answer rate: {results_df['false_answer_flag'].mean():.2%}")
        print(f"Average score: {results_df['score'].mean():.4f}")
        print(f"💾 Results saved to: {output_file}")
        print(f"{'='*80}\n")
        
        return results_df

    def run_experiment(self, dataset, model_name, exp_type, reward_correct, reward_abstain, reward_incorrect):
        """Main entry point for GPT experiments.
        
        Args:
            dataset: Pre-loaded dataset to run experiment on
            model_name: Name of the model to use
            exp_type: Type of experiment to run
            reward_correct: Reward for correct answers
            reward_abstain: Reward for abstaining
            reward_incorrect: Reward/penalty for incorrect answers
        """
        
        # Get model configuration
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_config = self.MODEL_CONFIGS[model_name]
        
        print(f"\n🚀 Running {exp_type.upper()} experiment (Non-batch mode)...")
        print("   Calling LLM one at a time for each question\n")
        
        try:
            return self.run_experiment_no_batch(
                dataset, exp_type, reward_correct, reward_abstain, reward_incorrect,
                model_name, model_config
            )
        except Exception as e:
            print(f"❌ Error during experiment: {str(e)}")
            import traceback
            traceback.print_exc()
