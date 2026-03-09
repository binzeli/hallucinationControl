import os
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
import glob

# Load environment variables from .env file
load_dotenv()

# ------------------------
# CONFIGURATION
# ------------------------


client = OpenAI(api_key=os.getenv("API_KEY"))
# MODEL = "gpt-4o-mini-2024-07-18"
MODEL = "gpt-5-mini-2025-08-07"

BATCH_SIZE = 5000 if "gpt-4o" in MODEL else 10000

# ------------------------
# HELPERS
# ------------------------

def extract_exp2_fields(text):
    """Parse single-response format; handles blank confidence and % signs."""
    text = text.strip().replace("\r\n", "\n")  # normalize newlines

    # Anchor to start of line (MULTILINE) to avoid matching 'Best Guess Confidence' for plain 'Confidence'
    ans_match       = re.search(r"(?im)^\s*(?:answer|response)\s*:\s*(.+?)\s*$", text)
    conf_match      = re.search(r"(?im)^\s*confidence\s*:\s*([0-9]+(?:\.[0-9]+)?)?\s*%?\s*$", text)
    best_match      = re.search(r"(?im)^\s*best\s*guess\s*:\s*(.+?)\s*$", text)
    best_conf_match = re.search(r"(?im)^\s*best\s*guess\s*confidence\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%?\s*$", text)

    answer = ans_match.group(1).strip() if ans_match else None
    conf = float(conf_match.group(1)) if (conf_match and conf_match.group(1)) else None
    best_guess = best_match.group(1).strip().rstrip(",") if best_match else None
    if best_guess and best_guess.lower() in {"", "none", "n/a"}:
        best_guess = None
    best_conf = float(best_conf_match.group(1)) if best_conf_match else None

    return answer, conf, best_guess, best_conf


def normalize(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation)).strip()

def is_correct(llm_answer, gold_answers):
    """Return True if llm_answer matches or is contained in any gold answer."""
    if all(isinstance(g, str) and len(g) == 1 for g in gold_answers):
        try:
            joined = "".join(gold_answers)
            gold_answers = ast.literal_eval(joined)
        except Exception:
            gold_answers = ["".join(gold_answers)]

    if llm_answer is None:
        return False
    norm_llm = normalize(llm_answer)
    for gold in gold_answers:
        norm_gold = normalize(gold)
        if norm_gold in norm_llm or norm_llm in norm_gold:
            return True
    return False


# ------------------------
# BATCH API FUNCTIONS
# ------------------------

def create_batch_input_file(dataset, start_idx, end_idx):
    """Create JSONL file with batch requests for a range of indices."""
    # Include model name in filename to avoid conflicts between different models
    model_short = "gpt5" if "gpt-5" in MODEL else "gpt4"
    input_file = f"main_exp/popQA/ablation_study/remove_reward/outputs/batch_files/batch_input_{model_short}_{start_idx}-{end_idx}.jsonl"
    
    os.makedirs(os.path.dirname(input_file), exist_ok=True)
    
    
    with open(input_file, 'w') as f:
        for idx in tqdm(range(start_idx, end_idx), desc=f"Creating batch input [{start_idx}-{end_idx}]"):
            ex = dataset[idx]
            q = ex["question"]
            
            prompt = (
                "Give your confidence about your answer in decimal number between 0 and 1, where 0 means you are completely unsure and 1 means you are completely sure. Use four decimal places for precision without unnecessary rounding (e.g., 0.8765).\n\n"
                "If you truly don't know, first say 'I don't know' and do not give a confidence for that (in that case just leave it blank). If you choose to answer, then provide your answer and your confidence for it\n"
                "Next, if you said 'I don't know', then in the same response, give your best possible guess for the answer and your confidence for it.\n\n"
                "Format your response as below such that you must begin your response by saying \"Answer:\" first:\n"
                "Answer: <your first answer>\n"
                "Confidence: <0–1>\n"
                "If you answered 'I don't know', then also include:\n"
                "Best Guess: <your best possible answer>\n"
                "Best Guess Confidence: <0–1>\n\n"
                f"Question: {q}"
            )
            
            batch_request = {
                "custom_id": f"request-{idx}",
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": MODEL,
                    "input": [
                        {"role": "user", "content": prompt}
                    ],
                    "reasoning": {
                        "effort": "minimal"
                    }
                    # "temperature": 0.0
                }
            }
            
            f.write(json.dumps(batch_request) + '\n')
    
    print(f"✅ Batch input file created: {input_file}")
    return input_file


def upload_batch_file(input_file):
    """Upload the batch input file to OpenAI."""
    print(f"📤 Uploading batch file: {input_file}")
    with open(input_file, 'rb') as f:
        batch_input_file = client.files.create(
            file=f,
            purpose="batch"
        )
    print(f"✅ File uploaded with ID: {batch_input_file.id}")
    return batch_input_file.id


def create_batch_job(input_file_id):
    """Create a batch processing job."""
    print(f"🚀 Creating batch job...")
    batch = client.batches.create(
        input_file_id=input_file_id,
        endpoint="/v1/responses",
        completion_window="24h"
    )
    print(f"✅ Batch job created with ID: {batch.id}")
    print(f"   Status: {batch.status}")
    return batch.id


def check_batch_status(batch_id):
    """Check the status of a batch job."""
    batch = client.batches.retrieve(batch_id)
    if batch.status == "failed" and hasattr(batch, 'errors'):
        print(f"   Batch Errors: {batch.errors}")
    return batch


def wait_for_batch_completion(batch_id, check_interval=60):
    """Wait for batch to complete, checking periodically."""
    print(f"⏳ Waiting for batch {batch_id} to complete...")
    print(f"   Checking status every {check_interval} seconds")
    
    while True:
        batch = check_batch_status(batch_id)
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


def download_batch_results(batch, dataset, start_idx, end_idx):
    """Download and process batch results."""
    if batch.status != "completed":
        print(f"❌ Batch is not completed. Status: {batch.status}")
        return None
    
    print(f"📥 Downloading batch results...")
    
    output_file_id = batch.output_file_id
    file_response = client.files.content(output_file_id)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"main_exp/popQA/ablation_study/remove_reward/outputs/batch_files/batch_output_{start_idx}-{end_idx}_{timestamp}.jsonl"
    
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
            gold = ex["possible_answers"]
            s_pop = ex.get("s_pop", None)
            o_pop = ex.get("o_pop", None)
            
            if result["response"]["status_code"] == 200:
                response_body = result["response"]["body"]
                out = None
                
                # GPT-5 batch format: response.body.output[1].content[0].text
                if "output" in response_body and isinstance(response_body["output"], list):
                    for output_item in response_body["output"]:
                        if output_item.get("type") == "message" and "content" in output_item:
                            for content_item in output_item["content"]:
                                if content_item.get("type") == "output_text" and "text" in content_item:
                                    out = content_item["text"]
                                    break
                            if out:
                                break
                # Fallback for other formats
                elif "output_text" in response_body:
                    out = response_body["output_text"]
                elif "choices" in response_body:
                    out = response_body["choices"][0]["message"]["content"]
                
                if not out:
                    print(f"⚠️  Could not extract text from {custom_id}")
                    continue
                
                ans, conf, best, best_conf = extract_exp2_fields(out)
                timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                idk_flag = bool(re.search(r"i don't know", ans or "", re.IGNORECASE))
                
                if idk_flag:
                    correct = is_correct(best, gold)
                else:
                    correct = is_correct(ans, gold)
                
                false_flag = int(not correct and not idk_flag)
                
                results.append({
                    "dataset_id": idx,
                    "timestamp": timestamp_str,
                    "s_pop": s_pop, "o_pop": o_pop,
                    "question": q, "possible_answers": gold,
                    "first_answer": ans, "first_confidence": conf,
                    "best_guess": best, "best_guess_confidence": best_conf,
                    "correct": int(correct), 
                    "idk_flag": int(idk_flag), "false_answer_flag": false_flag
                })
            else:
                print(f"⚠️  Error in request {custom_id}: {result['response']['status_code']}")
    
    return pd.DataFrame(results)


def run_single_batch(dataset, start_idx, end_idx):
    """Run a single batch experiment for a range of indices."""
    print(f"\n{'='*80}")
    print(f"Processing batch: {start_idx} to {end_idx}")
    print(f"{'='*80}")
    
    # Step 1: Create batch input file
    input_file = create_batch_input_file(dataset, start_idx, end_idx)
    
    # Step 2: Upload file
    input_file_id = upload_batch_file(input_file)
    
    # Step 3: Create batch job
    batch_id = create_batch_job(input_file_id)
    
    # Save batch info for tracking
    batch_info = {
        "batch_id": batch_id,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return batch_id, batch_info




def process_existing_batch_output(output_file, dataset, start_idx, end_idx):
    """Process an existing batch output file."""
    print(f"📥 Processing existing batch output: {output_file}")
    
    results = []
    with open(output_file, 'r') as f:
        for line in f:
            result = json.loads(line)
            custom_id = result["custom_id"]
            idx = int(custom_id.split("-")[1])
            
            ex = dataset[idx]
            q = ex["question"]
            gold = ex["possible_answers"]
            s_pop = ex.get("s_pop", None)
            o_pop = ex.get("o_pop", None)
            
            if result["response"]["status_code"] == 200:
                response_body = result["response"]["body"]
                out = None
                
                # GPT-5 batch format: response.body.output[1].content[0].text
                if "output" in response_body and isinstance(response_body["output"], list):
                    for output_item in response_body["output"]:
                        if output_item.get("type") == "message" and "content" in output_item:
                            for content_item in output_item["content"]:
                                if content_item.get("type") == "output_text" and "text" in content_item:
                                    out = content_item["text"]
                                    break
                            if out:
                                break
                # Fallback for other formats
                elif "output_text" in response_body:
                    out = response_body["output_text"]
                elif "choices" in response_body:
                    out = response_body["choices"][0]["message"]["content"]
                
                if not out:
                    continue
                
                ans, conf, best, best_conf = extract_exp2_fields(out)
                timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                idk_flag = bool(re.search(r"i don't know", ans or "", re.IGNORECASE))
                
                if idk_flag:
                    correct = is_correct(best, gold)
                else:
                    correct = is_correct(ans, gold)
                
                false_flag = int(not correct and not idk_flag)
                
                results.append({
                    "dataset_id": idx,
                    "timestamp": timestamp_str,
                    "s_pop": s_pop, "o_pop": o_pop,
                    "question": q, "possible_answers": gold,
                    "first_answer": ans, "first_confidence": conf,
                    "best_guess": best, "best_guess_confidence": best_conf,
                    "correct": int(correct),
                    "idk_flag": int(idk_flag), "false_answer_flag": false_flag
                })
    
    return pd.DataFrame(results)


def run_multi_batch_experiment(dataset, batch_size=BATCH_SIZE):
    """Run experiment in multiple batches sequentially, waiting for each to complete."""
    total_size = len(dataset)
    num_batches = (total_size + batch_size - 1) // batch_size
    
    print(f"📊 Total dataset size: {total_size}")
    print(f"📦 Batch size: {batch_size}")
    print(f"🔢 Number of batches needed: {num_batches}")
    print(f"⚠️  Processing sequentially: each batch waits for previous to complete\n")
    
    all_results = []
    batch_jobs = []
    tracking_file = f"main_exp/popQA/ablation_study/remove_reward/outputs/batch_files/batch_tracking.json"
    
    # Process batches sequentially
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_size)
        
        # Submit batch
        batch_id, batch_info = run_single_batch(dataset, start_idx, end_idx)
        batch_jobs.append(batch_info)
        
        # Save batch tracking file
        with open(tracking_file, 'w') as f:
            json.dump(batch_jobs, f, indent=2)
        
        print(f"✅ Batch {i+1}/{num_batches} submitted: {batch_id}")
        
        # Wait for this batch to complete before submitting the next one
        print(f"\n⏳ Waiting for batch {i+1}/{num_batches} to complete before submitting next batch...\n")
        batch = wait_for_batch_completion(batch_id)
        
        if batch.status == "completed":
            print(f"\n📥 Downloading results for batch {i+1}/{num_batches}...")
            results_df = download_batch_results(batch, dataset, start_idx, end_idx)
            if results_df is not None:
                all_results.append(results_df)
                print(f"✅ Batch {i+1}/{num_batches} processed successfully ({len(results_df)} results)")
            else:
                print(f"⚠️  Failed to process results for batch {i+1}/{num_batches}")
        else:
            print(f"❌ Batch {i+1}/{num_batches} failed with status: {batch.status}")
            print("⚠️  Stopping sequential processing due to failure")
            break
        
        # Small delay before next batch
        if i < num_batches - 1:
            print(f"\n{'='*80}")
            print(f"Moving to next batch...\n")
            time.sleep(5)
    
    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_df = combined_df.sort_values('dataset_id').reset_index(drop=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_file = f"main_exp/popQA/ablation_study/remove_reward/outputs/popqa_batch_results_{timestamp}.csv"
        combined_df.to_csv(final_file, index=False)
        
        print(f"\n{'='*80}")
        print(f"🎉 All batches processed successfully!")
        print(f"📊 Total results: {len(combined_df)}")
        print(f"💾 Final results saved to: {final_file}")
        print(f"📝 Batch tracking saved to: {tracking_file}")
        print(f"{'='*80}")
        
        return combined_df, batch_jobs
    else:
        print("\n❌ No results to combine")
        return None, batch_jobs



# ------------------------
# MAIN EXECUTION
# ------------------------
if __name__ == "__main__":
    import sys
    
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)

    print("📚 Loading PopQA (akariasai/popQA CSV)...")
    dataset = load_dataset("akariasai/PopQA")
    dataset = dataset["test"]

    # exp_type removed — filenames and functions no longer use it
    # Parse command line arguments
        # Check for sample size argument
    n_samples = None
    if len(sys.argv) > 1:
        try:
            n_samples = int(sys.argv[1])
            print(f"⚙️  Running with {n_samples} samples only")
            dataset = dataset.select(range(n_samples))
        except ValueError:
            print(f"⚠️  Invalid argument: {sys.argv[1]}")
            print("Usage: python popQA_B_batch_multi.py [n_samples|download]")
            print("Example: python popQA_B_batch_multi.py 10")
            sys.exit(1)
        
        # Submit batches sequentially
    print("\n🚀 Running Batch API ...")
    print("   Processing batches sequentially to avoid token limit\n")
        
    try:
        run_multi_batch_experiment(dataset, batch_size=BATCH_SIZE)
    except Exception as e:
        print(f"❌ Error during batch processing: {str(e)}")
        import traceback
        traceback.print_exc()
