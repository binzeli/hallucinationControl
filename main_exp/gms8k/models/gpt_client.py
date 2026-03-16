"""
GPT-specific implementation for running experiments with OpenAI models on GSM8K dataset.
Adapted from popQA gpt_client.py with numeric answer extraction from "####" lines.
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
import ast
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
    Runner for GPT experiments using OpenAI's Batch API on GSM8K dataset.
    """
    
    # Model configurations
    MODEL_CONFIGS = {
        "gpt-4o-mini": {
            "full_name": "gpt-4o-mini-2024-07-18",
            "batch_size": 5000,  # 2M token limit
            "short_name": "gpt4"
        },
        "gpt-5-mini": {
            "full_name": "gpt-5-mini-2025-08-07",
            "batch_size": 10000,  # 5M token limit
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
    # BATCH API FUNCTIONS
    # ------------------------



    def create_batch_input_file(self, dataset, start_idx, end_idx, exp_type, reward_correct, reward_abstain, 
                                reward_incorrect, model_name, model_config):
        """Create JSONL file with batch requests for a range of indices."""
        if exp_type not in self.experiments:
            raise ValueError(f"Unknown experiment type: {exp_type}")
        
        exp_config = self.experiments[exp_type]
        input_file = f"main_exp/gms8k/outputs/batch_files/batch_input_{exp_type}_{model_config['short_name']}_{start_idx}-{end_idx}.jsonl"
        
        os.makedirs(os.path.dirname(input_file), exist_ok=True)
        
        with open(input_file, 'w') as f:
            for idx in tqdm(range(start_idx, end_idx), desc=f"Creating batch input [{start_idx}-{end_idx}]"):
                ex = dataset[idx]
                q = ex["question"]
                
                prompt = get_experiment_prompt(
                    reward_correct=reward_correct,
                    reward_abstain=reward_abstain,
                    reward_incorrect=reward_incorrect,
                    question=q,
                    exp_type=exp_type
                )
                
                batch_request = {
                    "custom_id": f"request-{idx}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model_config["full_name"],
                        "messages": []
                    }
                }
                
                # Set temperature based on model
                if "gpt-4" in model_name:
                    batch_request["body"]["temperature"] = 0
                
                # Add reasoning parameter only for GPT-5 models
                if "gpt-5" in model_name:
                    batch_request["body"]["reasoning"] = {"effort": "minimal"}
                
                if exp_config["use_system_prompt"]:
                    batch_request["body"]["messages"].append({"role": "system", "content": SYSTEM_PROMPT})
                
                batch_request["body"]["messages"].append({"role": "user", "content": prompt})
                
                f.write(json.dumps(batch_request) + '\n')
        
        print(f"✅ Batch input file created: {input_file}")
        return input_file

    def upload_batch_file(self, input_file):
        """Upload the batch input file to OpenAI."""
        print(f"📤 Uploading batch file: {input_file}")
        with open(input_file, 'rb') as f:
            batch_input_file = self.client.files.create(file=f, purpose="batch")
        print(f"✅ File uploaded with ID: {batch_input_file.id}")
        return batch_input_file.id

    def create_batch_job(self, input_file_id):
        """Create a batch processing job."""
        print(f"🚀 Creating batch job...")
        batch = self.client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        print(f"✅ Batch job created with ID: {batch.id}")
        print(f"   Status: {batch.status}")
        return batch.id

    def check_batch_status(self, batch_id):
        """Check the status of a batch job."""
        batch = self.client.batches.retrieve(batch_id)
        if batch.status == "failed" and hasattr(batch, 'errors'):
            print(f"   Batch Errors: {batch.errors}")
        return batch

    def wait_for_batch_completion(self, batch_id, check_interval=60):
        """Wait for batch to complete, checking periodically."""
        print(f"⏳ Waiting for batch {batch_id} to complete...")
        print(f"   Checking status every {check_interval} seconds")
        
        while True:
            batch = self.check_batch_status(batch_id)
            completed = batch.request_counts.completed if hasattr(batch, 'request_counts') else 0
            total = batch.request_counts.total if hasattr(batch, 'request_counts') else 0
            print(f"   Status: {batch.status} | Completed: {completed}/{total}")
            
            if batch.status == "completed":
                print(f"✅ Batch completed!")
                return batch
            elif batch.status == "failed":
                print(f"❌ Batch failed!")
                return batch
            elif batch.status in ["expired", "cancelled"]:
                print(f"⚠️  Batch {batch.status}!")
                return batch
            
            time.sleep(check_interval)

    def download_batch_results(self, batch, dataset, start_idx, end_idx, exp_type, reward_correct, 
                               reward_abstain, reward_incorrect, model_config):
        """Download and process batch results."""
        if batch.status != "completed":
            print(f"❌ Batch is not completed. Status: {batch.status}")
            return None
        
        # Check if there are results
        if not hasattr(batch, 'output_file_id') or batch.output_file_id is None:
            print(f"⚠️  No output file available. Batch may have had errors or no successful requests.")
            print(f"   Batch status: {batch.status}")
            if hasattr(batch, 'error_file_id') and batch.error_file_id:
                print(f"   Error file ID: {batch.error_file_id}")
            return pd.DataFrame()  # Return empty dataframe
        
        print(f"📥 Downloading batch results...")
        
        output_file_id = batch.output_file_id
        file_response = self.client.files.content(output_file_id)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"main_exp/gms8k/outputs/batch_files/batch_output_{exp_type}_{model_config['short_name']}_{start_idx}-{end_idx}_{timestamp}.jsonl"
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'wb') as f:
            f.write(file_response.content)
        
        print(f"✅ Batch output saved to: {output_file}")
        
        # Process results
        results = []
        with open(output_file, 'r') as f:
            for line in f:
                result = json.loads(line)
                custom_id = result["custom_id"]
                idx = int(custom_id.split("-")[1])
                
                ex = dataset[idx]
                q = ex["question"]
                # Extract gold answer from "####" line
                gold_answer_text = ex["answer"]
                gold_answer = self.extract_numeric_answer_from_gsm8k(gold_answer_text)
                
                if result["response"]["status_code"] == 200:
                    response_body = result["response"]["body"]
                    out = None
                    
                    # GPT-5 batch format
                    if "output" in response_body and isinstance(response_body["output"], list):
                        for output_item in response_body["output"]:
                            if output_item.get("type") == "message" and "content" in output_item:
                                for content_item in output_item["content"]:
                                    if content_item.get("type") == "output_text" and "text" in content_item:
                                        out = content_item["text"]
                                        break
                                if out:
                                    break
                    elif "output_text" in response_body:
                        out = response_body["output_text"]
                    elif "choices" in response_body:
                        out = response_body["choices"][0]["message"]["content"]
                    
                    if not out:
                        print(f"⚠️  Could not extract text from {custom_id}")
                        continue
                    
                    ans, conf, best, best_conf = extract_fields_gsm8k(out)
                    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    idk_flag = bool(re.search(r"i don't know", out or "", re.IGNORECASE))
                    
                    # For GSM8K, if IDK flag is set, use best_guess, otherwise use first answer
                    if idk_flag:
                        llm_answer = best
                        correct = self.is_correct_gsm8k(llm_answer, gold_answer)
                    else:
                        llm_answer = ans
                        correct = self.is_correct_gsm8k(llm_answer, gold_answer)
                    
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
                else:
                    print(f"⚠️  Error in request {custom_id}: {result['response']['status_code']}")
        
        return pd.DataFrame(results)

    def run_single_batch(self, dataset, start_idx, end_idx, exp_type, reward_correct, reward_abstain, 
                         reward_incorrect, model_name, model_config):
        """Run a single batch experiment."""
        print(f"\n{'='*80}")
        print(f"Processing batch: {start_idx} to {end_idx}")
        print(f"{'='*80}")
        
        input_file = self.create_batch_input_file(dataset, start_idx, end_idx, exp_type, reward_correct, 
                                                   reward_abstain, reward_incorrect, model_name, model_config)
        input_file_id = self.upload_batch_file(input_file)
        batch_id = self.create_batch_job(input_file_id)
        
        batch_info = {
            "batch_id": batch_id,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return batch_id, batch_info

    def run_multi_batch_experiment(self, dataset, exp_type, reward_correct, reward_abstain, reward_incorrect, 
                                   model_name, model_config):
        """Run experiment in multiple batches sequentially."""
        if exp_type not in self.experiments:
            raise ValueError(f"Unknown experiment type: {exp_type}")
        
        exp_config = self.experiments[exp_type]
        total_size = len(dataset)
        # Use smaller batch size if dataset is smaller than max batch size
        batch_size = min(total_size, model_config["batch_size"])
        num_batches = (total_size + batch_size - 1) // batch_size
 
        all_results = []
        batch_jobs = []
        tracking_file = f"main_exp/gms8k/outputs/batch_files/batch_tracking_{exp_type}.json"
        
        os.makedirs(os.path.dirname(tracking_file), exist_ok=True)
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_size)
            
            batch_id, batch_info = self.run_single_batch(dataset, start_idx, end_idx, exp_type, 
                                                          reward_correct, reward_abstain, reward_incorrect,
                                                          model_name, model_config)
            batch_jobs.append(batch_info)
            
            with open(tracking_file, 'w') as f:
                json.dump(batch_jobs, f, indent=2)
            
            print(f"✅ Batch {i+1}/{num_batches} submitted: {batch_id}")
            
            print(f"\n⏳ Waiting for batch {i+1}/{num_batches} to complete before submitting next batch...\n")
            batch = self.wait_for_batch_completion(batch_id)
            
            if batch.status == "completed":
                print(f"\n📥 Downloading results for batch {i+1}/{num_batches}...")
                results_df = self.download_batch_results(batch, dataset, start_idx, end_idx, exp_type, 
                                                         reward_correct, reward_abstain, reward_incorrect, model_config)
                if results_df is not None and len(results_df) > 0:
                    all_results.append(results_df)
                    print(f"✅ Batch {i+1}/{num_batches} processed successfully ({len(results_df)} results)")
                elif results_df is not None and len(results_df) == 0:
                    print(f"⚠️  Batch {i+1}/{num_batches} completed but returned no results")
                else:
                    print(f"⚠️  Failed to process results for batch {i+1}/{num_batches}")
            else:
                print(f"❌ Batch {i+1}/{num_batches} failed with status: {batch.status}")
                print("⚠️  Stopping sequential processing due to failure")
                break
            
            if i < num_batches - 1:
                print(f"\n{'='*80}")
                print(f"Moving to next batch...\n")
                time.sleep(5)
        
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
            final_file = f"main_exp/gms8k/outputs/{model_config['short_name']}_results/gsm8k_{exp_type}_results_{reward_str}_{timestamp}.csv"
            
            os.makedirs(os.path.dirname(final_file), exist_ok=True)
            combined_df.to_csv(final_file, index=False)
            
            print(f"\n{'='*80}")
            print(f"🎉 All batches processed successfully!")
            print(f"📊 Total results: {len(combined_df)}")
            print(f"Accuracy: {combined_df['correct'].mean():.2%}")
            print(f"IDK rate: {combined_df['idk_flag'].mean():.2%}")
            print(f"False answer rate: {combined_df['false_answer_flag'].mean():.2%}")
            print(f"Average score: {combined_df['score'].mean():.4f}")
            print(f"💾 Final results saved to: {final_file}")
            print(f"📝 Batch tracking saved to: {tracking_file}")
            print(f"{'='*80}")
            
            return combined_df, batch_jobs
        else:
            print("\n❌ No results collected!")
            return None, batch_jobs

    def run_experiment(self, dataset, model_name, exp_type, reward_correct, reward_abstain, reward_incorrect):
        """Run the full experiment."""
        model_config = self.MODEL_CONFIGS[model_name]
        print(f"🎯 Using model configuration for {model_name}:")
        print(f"   Full name: {model_config['full_name']}")
        print(f"   Batch size: {model_config['batch_size']}\n")
        
        return self.run_multi_batch_experiment(dataset, exp_type, reward_correct, reward_abstain, 
                                               reward_incorrect, model_name, model_config)
