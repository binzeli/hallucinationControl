"""
MMLU-Pro Async API runner.
Includes token-logprob aggregation (key-token) and no-CoT format (option + confidence on last line).

Outputs CSV columns:
index, question, options, correct_answer,
model_answers, model_answer, answer_distribution,
self_confidences, api_confidences_key_token
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import re
import time
import math
import uuid
import asyncio
from datetime import datetime

import pandas as pd
import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv

from utils.yaml_parser import load_yaml
from utils.api_caller import get_responses_concurrently_with_ids

# Load environment variables
load_dotenv()

# ============================================================================
# Custom processing functions for no-CoT format
# Format: First line(s) = Thinking/Reasoning, Last line = "A 0.8500"
# ============================================================================

def extract_answer_and_confidence_no_cot(response_text):
    """
    Extract answer and verbal confidence from no-CoT format response.
    Expected format:
    - First line(s): Thinking/Reasoning text
    - Last line: "A 0.8500" or "B 0.9500"

    Returns: (answer_letter, confidence) or (None, None) if not found
    """
    if not response_text:
        return None, None

    # Remove "Response:" prefix if present
    response_text = re.sub(r'^Response:\s*', '', response_text, flags=re.IGNORECASE).strip()

    # Split into lines
    lines = [line.strip() for line in response_text.split('\n') if line.strip()]
    if not lines:
        return None, None

    # Last line should contain answer and confidence (e.g., "B 0.9500")
    last_line = lines[-1]
    words = last_line.split()

    # Try pattern: last word is confidence, second-to-last is option
    if len(words) >= 2:
        try:
            last_word = words[-1]
            confidence = float(last_word)
            if 0.0 <= confidence <= 1.0:
                option = re.sub(r'[^\w]', '', words[-2].strip())
                if len(option) == 1 and option.isalpha() and option.isupper():
                    return option.upper(), confidence
        except (ValueError, IndexError):
            pass

    # Try pattern: search backward for "X Y"
    for i in range(len(words) - 1, 0, -1):
        try:
            confidence = float(words[i])
            if 0.0 <= confidence <= 1.0:
                option = re.sub(r'[^\w]', '', words[i - 1].strip())
                if len(option) == 1 and option.isalpha() and option.isupper():
                    return option.upper(), confidence
        except (ValueError, IndexError):
            continue

    return None, None


def process_response_no_cot(raw_response, include_log_prob=True):
    """
    Process raw response for no-CoT format.
    Format: "Thinking: ...\\nA 0.8500"
    """
    if not raw_response or "choices" not in raw_response or len(raw_response["choices"]) == 0:
        return None

    content = raw_response["choices"][0]["message"]["content"]

    answer_letter, self_confidence = extract_answer_and_confidence_no_cot(content)
    if answer_letter is None or self_confidence is None:
        print(f"Could not extract answer/confidence from: {content[:100]}...")
        return None

    result = {
        "model_answer": answer_letter,
        "self_confidence": self_confidence
    }

    if include_log_prob and "logprobs" in raw_response["choices"][0]:
        logprobs_obj = raw_response["choices"][0]["logprobs"]

        if logprobs_obj and "content" in logprobs_obj:
            tokens_with_logprobs = logprobs_obj["content"]

            # Find option token logprob
            option_token_logprob = None
            for token_info in tokens_with_logprobs:
                token_text = token_info.get("token", "").strip()
                cleaned = re.sub(r'[\s\n\t]', '', token_text)
                if cleaned == answer_letter:
                    option_token_logprob = token_info.get("logprob", None)
                    if option_token_logprob is not None:
                        break

            result["api_confidence_key_token"] = math.exp(option_token_logprob) if option_token_logprob is not None else None
        else:
            result["api_confidence_key_token"] = None
    else:
        result["api_confidence_key_token"] = None

    return result


async def repeat_call_single_question(row, prompt_data, api_key, api_url, num_repeats=10, include_log_prob=True):
    """
    Repeat API calls for a single question multiple times to test robustness.
    """
    results = []
    model = prompt_data["parameters"]["model"]
    temperature = prompt_data["parameters"].get("temperature", 0.0)

    question = row["question"]
    options = row["options"]

    labeled_options = [f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)]
    formatted_options = "\n".join(labeled_options)
    prompt = prompt_data["prompt"].format(question=question, options=formatted_options)

    question_id = row.get("question_id", row.name)
    print(f"Repeating API calls for question_id {question_id} {num_repeats} times...")
    print(f"Using temperature: {temperature}")

    batch_size = 5
    for batch_start in range(0, num_repeats, batch_size):
        batch_end = min(batch_start + batch_size, num_repeats)

        batch_payloads = []
        batch_request_ids = []
        batch_contexts = {}

        for run_number in range(batch_start, batch_end):
            request_id = str(uuid.uuid4())
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "logprobs": include_log_prob,
                "temperature": temperature,
            }
            batch_payloads.append(payload)
            batch_request_ids.append(request_id)
            batch_contexts[request_id] = {
                "request_id": request_id,
                "run_number": run_number + 1,  # 1-indexed
                "index": question_id,
                "question": question,
                "options": labeled_options,
                "correct_answer": row["answer"],
            }

        responses = await get_responses_concurrently_with_ids(
            api_url, batch_payloads, batch_request_ids, api_key,
            include_log_prob=include_log_prob, return_raw_response=True
        )

        for response in responses:
            if not response:
                continue

            request_id = response.get("request_id")
            context = batch_contexts.get(request_id)
            if not context:
                continue

            if "error" in response:
                error_msg = response.get("error", "Unknown error")
                print(f"Error in response for run {context['run_number']}: {error_msg}")
                with open("error_responses.log", "a") as f:
                    f.write(
                        f"Run Number: {context['run_number']}\n"
                        f"Request ID: {request_id}\n"
                        f"Context: {context}\n"
                        f"Error: {error_msg}\n\n"
                    )
                continue

            try:
                raw_response = response.get("raw_response")
                if not raw_response:
                    print(f"No raw response for run {context['run_number']}")
                    continue

                processed = process_response_no_cot(raw_response, include_log_prob=include_log_prob)
                if not processed:
                    print(f"Failed to process response for run {context['run_number']}")
                    continue

                model_answer = processed.get("model_answer")
                self_confidence = processed.get("self_confidence")

                if model_answer is None or self_confidence is None:
                    print(f"Incomplete response data for run {context['run_number']}")
                    continue

                result = {
                    "run_number": context["run_number"],
                    "index": context["index"],
                    "question": context["question"],
                    "options": "\n".join(context["options"]),
                    "correct_answer": context["correct_answer"],
                    "model_answer": model_answer,
                    "self_confidence": self_confidence,
                }

                if include_log_prob:
                    result["api_confidence_key_token"] = processed.get("api_confidence_key_token")

                results.append(result)
                print(
                    f"Run {context['run_number']}/{num_repeats} completed. "
                    f"Answer: {model_answer}, Self-conf: {self_confidence:.4f}"
                )
            except Exception as e:
                print(f"Error processing response for run {context['run_number']}: {e}")
                import traceback
                traceback.print_exc()
                with open("error_responses.log", "a") as f:
                    f.write(
                        f"Run Number: {context['run_number']}\n"
                        f"Context: {context}\n"
                        f"Error: {e}\n\n"
                    )

        if batch_end < num_repeats:
            await asyncio.sleep(1)

    return results


async def process_dataset_with_repeats(dataset, prompt_data, api_key, api_url, num_repeats=10, include_log_prob=True, batch_size=5):
    """
    Process the entire dataset, repeating each question multiple times and computing statistics.
    """
    aggregated_results = []
    total_questions = len(dataset)

    print(f"Processing {total_questions} questions, {num_repeats} repeats each...")

    question_counter = 0
    for question_idx in range(0, total_questions, batch_size):
        batch_end = min(question_idx + batch_size, total_questions)
        batch_dataset = dataset.iloc[question_idx:batch_end]

        print(f"\nProcessing questions {question_idx+1}-{batch_end} of {total_questions}...")

        for _, row in batch_dataset.iterrows():
            question_counter += 1
            question_id = row.get("question_id", row.name)
            print(f"Processing question {question_id} ({question_counter}/{total_questions})...")

            repeat_results = await repeat_call_single_question(
                row, prompt_data, api_key, api_url,
                num_repeats=num_repeats,
                include_log_prob=include_log_prob
            )

            if not repeat_results:
                print(f"Warning: No results for question {question_id}. Skipping.")
                continue

            repeat_df = pd.DataFrame(repeat_results)

            if "run_number" in repeat_df.columns:
                repeat_df = repeat_df.sort_values("run_number").reset_index(drop=True)

            aggregated_result = {
                "index": question_id,
                "question": row["question"],
                "options": "\n".join([f"{chr(65 + i)}. {opt}" for i, opt in enumerate(row["options"])]),
                "correct_answer": row["answer"],
            }

            num_repeats_stored = 0
            if "model_answer" in repeat_df.columns:
                model_answers = repeat_df["model_answer"].tolist()
                aggregated_result["model_answers"] = ";".join([str(ans) for ans in model_answers])
                num_repeats_stored = len(model_answers)

                most_common_answer = repeat_df["model_answer"].mode()
                aggregated_result["model_answer"] = (
                    most_common_answer.iloc[0]
                    if len(most_common_answer) > 0
                    else repeat_df["model_answer"].iloc[0]
                )

                answer_counts = repeat_df["model_answer"].value_counts().sort_index()
                aggregated_result["answer_distribution"] = ", ".join([f"{ans}:{count}" for ans, count in answer_counts.items()])

            if "self_confidence" in repeat_df.columns:
                self_conf_values = repeat_df["self_confidence"].dropna().tolist()
                aggregated_result["self_confidences"] = ";".join([f"{val:.6f}" for val in self_conf_values])

            if include_log_prob:
                if "api_confidence_key_token" in repeat_df.columns:
                    vals = repeat_df["api_confidence_key_token"].dropna().tolist()
                    aggregated_result["api_confidences_key_token"] = ";".join([f"{val:.6f}" for val in vals])

            aggregated_results.append(aggregated_result)
            print(f"Question {question_id} completed. Stored {num_repeats_stored} repeats.")
            print(f"  Answer distribution: {aggregated_result.get('answer_distribution', 'N/A')}")

        if batch_end < total_questions:
            await asyncio.sleep(2)

    return aggregated_results


async def main(category, prompt_file, output_file, api_key, api_url, results_folder, output_base="MMLU-Pro", batch_size=5, num_repeats=10, num_questions=None):
    """
    Main function to run the experiment.
    """
    print("=" * 60)
    print(f"{category.capitalize()} No-CoT Experiment")
    print("=" * 60)

    print("Loading MMLU-Pro dataset...")
    ds = load_dataset("TIGER-Lab/MMLU-Pro")
    df = ds["test"].to_pandas()

    print(f"Total dataset size: {len(df)}")
    print(f"Target category: {category}")

    if "subject" in df.columns:
        category_df = df[df["subject"] == category].copy()
        print(f"{category.capitalize()} dataset size (filtered by 'subject'): {len(category_df)}")
    elif "category" in df.columns:
        category_df = df[df["category"] == category].copy()
        print(f"{category.capitalize()} dataset size (filtered by 'category'): {len(category_df)}")
    else:
        print("Warning: No 'subject' or 'category' column found. Using all data.")
        category_df = df.copy()

    if category_df is None or len(category_df) == 0:
        print(f"Error: Could not find {category} data. Exiting.")
        sys.exit(1)

    if num_questions is not None and num_questions > 0:
        category_df = category_df.head(num_questions).copy()
        print(f"Limited to first {len(category_df)} {category} questions")
    else:
        print(f"Running on full {category} dataset: {len(category_df)} questions")

    # Prompt file path based on this script location (prompts/ is inside preliminary_exp/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(script_dir, "prompts", category, prompt_file)

    if not os.path.exists(prompt_path):
        print(f"Error: Prompt file not found at {prompt_path}")
        print(f"Please check that the prompt file exists in prompts/{category}/ directory.")
        sys.exit(1)

    prompt_path = os.path.abspath(prompt_path)
    prompt_data = load_yaml(prompt_path)
    print(f"Loaded prompt from: {prompt_path}")

    start_time = time.time()

    results = await process_dataset_with_repeats(
        category_df, prompt_data, api_key, api_url,
        num_repeats=num_repeats,
        include_log_prob=True,
        batch_size=batch_size
    )

    elapsed_time = time.time() - start_time
    print(f"\nProcessing completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes).")
    print(f"Successfully processed {len(results)} questions.")

    os.makedirs(results_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if '.' in output_file:
        name, ext = output_file.rsplit('.', 1)
        output_path = os.path.join(results_folder, f"{output_base}-{name}-{timestamp}.{ext}")
    else:
        output_path = os.path.join(results_folder, f"{output_base}-{output_file}-{timestamp}.csv")

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}.")

    return results_df


if __name__ == "__main__":
    CATEGORY = "biology"
    PROMPT_FILE = "biology-1.yaml"
    OUTPUT_FILE = "biology-1"
    OUTPUT_BASE = "MMLU-Pro"  # Prefix to distinguish from PopQA results
    API_URL = "https://api.openai.com/v1/chat/completions"
    API_KEY_ENV = "API_KEY_PROJ"
    RESULTS_FOLDER = "preliminary_exp/outputs"
    BATCH_SIZE = 5  # Number of questions per batch
    NUM_REPEATS = 10
    NUM_QUESTIONS = 50

    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        print(f"Error: API key not found. Set {API_KEY_ENV} in your environment or .env file.")
        sys.exit(1)
    print(f"Using API key from environment variable: {API_KEY_ENV}")

    results_df = asyncio.run(
        main(
            category=CATEGORY,
            prompt_file=PROMPT_FILE,
            output_file=OUTPUT_FILE,
            api_key=api_key,
            api_url=API_URL,
            results_folder=RESULTS_FOLDER,
            output_base=OUTPUT_BASE,
            batch_size=BATCH_SIZE,
            num_repeats=NUM_REPEATS,
            num_questions=NUM_QUESTIONS
        )
    )
