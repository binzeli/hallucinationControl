"""
PopQA Async API runner.
Includes token-logprob aggregation (avg, min) and handles IDK -> Best Guess logic.

Outputs CSV columns:
dataset_id, timestamp, s_pop, o_pop, question, possible_answers, 
first_answer, first_confidence, best_guess, best_guess_confidence,
correct, score, idk_flag, false_answer_flag,
self_confidence, api_confidence_min, api_confidence_avg
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import re
import time
import math
import ast
import uuid
import asyncio
import string
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv

from utils.api_caller import get_responses_concurrently_with_ids

# Load environment variables
load_dotenv()

# ============================================================================
# PROMPT
# ============================================================================

def get_prompt(q):
    """Generate prompt without incentive descriptions, supporting IDK and Best Guess."""
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
    return prompt

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_exp2_fields(text):
    """Parse response format; handles blank confidence and % signs."""
    if not text:
        return None, None, None, None
    text = text.strip().replace("\r\n", "\n")

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


def confidence_to_01(value, scale=1):
    """Map parsed confidence to [0, 1] for analysis.
    scale: 1 = [0,1], 100 = [0,100], 'percent'/'pct' = 0%-100% (parsed as 0-100).
    """
    if value is None or (isinstance(value, float) and (value != value)):  # NaN
        return None
    v = float(value)
    if scale == 1:
        pass
    elif scale == 100 or scale in ("percent", "pct"):
        v = v / 100.0
    else:
        pass  # treat as [0,1]
    return max(0.0, min(1.0, v))


def normalize(text):
    """Normalize text for comparison."""
    return text.lower().translate(str.maketrans('', '', string.punctuation)).strip()

def is_correct(llm_answer, gold_answers):
    """Return True if llm_answer matches any gold answer."""
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

def find_answer_tokens_start_index(tokens_with_logprobs, idk_flag=False):
    """
    Find the start and end indices for answer tokens.
    If idk_flag is True, we look for tokens between "Best Guess:" and "Best Guess Confidence:".
    If idk_flag is False, we look for tokens between "Answer:" and "Confidence:".
    """
    if not tokens_with_logprobs:
        return None, None

    full_text = "".join([t.get("token", "") for t in tokens_with_logprobs])
    
    if idk_flag:
        start_pattern = r'(?i)best\s*guess\s*:'
        end_pattern = r'(?i)best\s*guess\s*confidence\s*:'
    else:
        start_pattern = r'(?i)answer\s*:'
        end_pattern = r'(?i)confidence\s*:'

    start_match = re.search(start_pattern, full_text)
    end_match = re.search(end_pattern, full_text)

    if not start_match or not end_match:
        return None, None

    start_char = start_match.end()
    end_char = end_match.start()

    start_idx = None
    end_idx = None
    char_pos = 0
    for i, token_info in enumerate(tokens_with_logprobs):
        token_len = len(token_info.get("token", ""))
        if start_idx is None and char_pos >= start_char:
            start_idx = i
        if end_idx is None and char_pos >= end_char:
            end_idx = i
            break
        char_pos += token_len

    return start_idx, end_idx

def process_response_with_logprobs(raw_response, idk_flag, include_log_prob=True):
    """Extract logprobs for the relevant answer part (Answer or Best Guess). Uses /v1/chat/completions format."""
    tokens_with_logprobs = None

    if "choices" in raw_response and len(raw_response["choices"]) > 0:
        if "logprobs" in raw_response["choices"][0]:
            logprobs_obj = raw_response["choices"][0]["logprobs"]
            if logprobs_obj and "content" in logprobs_obj:
                tokens_with_logprobs = logprobs_obj["content"]

    if not tokens_with_logprobs:
        return None, None

    start_idx, end_idx = find_answer_tokens_start_index(tokens_with_logprobs, idk_flag)
    
    if start_idx is not None and end_idx is not None:
        answer_tokens = tokens_with_logprobs[start_idx:end_idx]
    else:
        # Fallback
        answer_tokens = tokens_with_logprobs

    logprobs = [t.get("logprob") for t in answer_tokens if t.get("logprob") is not None]
    
    if not logprobs:
        return None, None

    api_avg = math.exp(sum(logprobs) / len(logprobs))
    api_min = math.exp(min(logprobs))
    return api_avg, api_min

async def process_dataset_popQA(dataset, api_key, model, batch_size, confidence_scale, api_url):
    results = []
    total_questions = len(dataset)

    for question_idx in range(0, total_questions, batch_size):
        batch_end = min(question_idx + batch_size, total_questions)
        batch_dataset = dataset.select(range(question_idx, batch_end))

        print(f"\nProcessing {question_idx+1}-{batch_end} of {total_questions}...")

        batch_payloads = []
        batch_request_ids = []
        batch_contexts = {}

        for i, ex in enumerate(batch_dataset):
            q = ex["question"]
            gold = ex["possible_answers"]
            dataset_id = question_idx + i

            prompt = get_prompt(q)
            request_id = str(uuid.uuid4())
            
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "logprobs": True,
                "temperature": 0.0,
            }
            batch_payloads.append(payload)
            batch_request_ids.append(request_id)
            batch_contexts[request_id] = {
                "dataset_id": dataset_id,
                "question": q,
                "possible_answers": gold,
                "s_pop": ex.get("s_pop"),
                "o_pop": ex.get("o_pop"),
            }

        responses = await get_responses_concurrently_with_ids(
            api_url, batch_payloads, batch_request_ids, api_key,
            include_log_prob=True, return_raw_response=True
        )

        for response in responses:
            if not response or "error" in response:
                continue

            request_id = response.get("request_id")
            ctx = batch_contexts.get(request_id)
            raw = response.get("raw_response")
            
            # Extract content text
            content = raw["choices"][0]["message"]["content"]
            print(f"Content: {content}")
            
            ans, conf, best, best_conf = extract_exp2_fields(content)
            
            # Map to [0,1] for analysis (prompt may ask 0-10 or 0-100)
            conf_01 = confidence_to_01(conf, confidence_scale)
            best_conf_01 = confidence_to_01(best_conf, confidence_scale)

            idk_flag = bool(re.search(r"i don't know", content or "", re.IGNORECASE))

            # Correctness and Score
            if idk_flag:
                correct = is_correct(best, ctx["possible_answers"])
                self_conf = best_conf_01
            else:
                correct = is_correct(ans, ctx["possible_answers"])
                self_conf = conf_01

            score = 1 if correct else 0
            false_flag = int(not correct and not idk_flag)

            # Token-wise confidence (avg and min: geometric mean and exp(min logprob))
            api_avg, api_min = process_response_with_logprobs(raw, idk_flag) or (None, None)

            if idk_flag:
                answer_display = best
                conf_str = f"{best_conf_01:.4f}" if best_conf_01 is not None else "N/A"
                print(
                    f"Question {ctx['dataset_id']} completed. "
                    f"IDK: yes, Best Guess: {answer_display}, Best Guess Confidence: {conf_str}"
                )
            else:
                answer_display = ans
                conf_str = f"{self_conf:.4f}" if self_conf is not None else "N/A"
                print(
                    f"Question {ctx['dataset_id']} completed. "
                    f"IDK: no, Answer: {answer_display}, Self-conf: {conf_str}"
                )

            results.append({
                "dataset_id": ctx["dataset_id"],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "s_pop": ctx["s_pop"],
                "o_pop": ctx["o_pop"],
                "question": ctx["question"],
                "possible_answers": str(ctx["possible_answers"]),
                "first_answer": ans,
                "first_confidence": conf_01,
                "best_guess": best,
                "best_guess_confidence": best_conf_01,
                "correct": int(correct),
                "score": score,
                "idk_flag": int(idk_flag),
                "false_answer_flag": false_flag,
                "self_confidence": self_conf,
                "api_confidence_min": api_min,
                "api_confidence_avg": api_avg
            })

    return results

async def main(api_key, model, batch_size, confidence_scale, output_base, results_folder, api_url, num_questions=None):
    """
    Main function to run the PopQA experiment.
    """
    print("=" * 60)
    print("PopQA Experiment")
    print("=" * 60)

    print("Loading PopQA dataset...")
    ds = load_dataset("akariasai/PopQA")["test"]
    if num_questions is not None and num_questions > 0:
        ds = ds.select(range(num_questions))
        print(f"Limited to first {num_questions} questions")
    print(f"Dataset size: {len(ds)}")

    results = await process_dataset_popQA(
        ds, api_key, model, batch_size, confidence_scale, api_url
    )

    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(results_folder, exist_ok=True)
    out_path = os.path.join(results_folder, f"{output_base}-{timestamp}.csv")
    df.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")

    return df


if __name__ == "__main__":
    MODEL = "gpt-4o-mini-2024-07-18"
    API_URL = "https://api.openai.com/v1/chat/completions"
    API_KEY_ENV = "API_KEY_PROJ"
    CONCURRENT_BATCH_SIZE = 10
    CONFIDENCE_SCALE = 1  # 1 = [0,1], 100 = [0,100], 'percent'/'pct' = 0%-100% (parsed as 0-100). Stored in CSV as [0,1].
    OUTPUT_BASE = "popQA"
    RESULTS_FOLDER = "preliminary_exp/outputs"
    NUM_QUESTIONS = None  # Set to int to limit (e.g. 10 for testing)

    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        print(f"Error: API key not found. Set {API_KEY_ENV} in your environment or .env file.")
        sys.exit(1)
    print(f"Using API key from environment variable: {API_KEY_ENV}")

    asyncio.run(
        main(
            api_key=api_key,
            model=MODEL,
            batch_size=CONCURRENT_BATCH_SIZE,
            confidence_scale=CONFIDENCE_SCALE,
            output_base=OUTPUT_BASE,
            results_folder=RESULTS_FOLDER,
            api_url=API_URL,
            num_questions=NUM_QUESTIONS
        )
    )

