# Preliminary Experiment: Validity of Self-Reported Confidence

Code for the preliminary experiment validating self-reported (verbal) confidence against API-derived token-level confidence. Uses two datasets: **PopQA** (open-ended QA with optional abstention) and **MMLU-Pro** (multiple-choice with repeated sampling).

Before running the scripts, create an `outputs/` directory under `preliminary_exp`; all result CSVs will be written there by the `run_*.py` scripts.

---

## PopQA

### Script: `run_PopQA.py`

**Parameters** (edit in `__main__`):


| Parameter               | Default                                      | Description                                 |
| ----------------------- | -------------------------------------------- | ------------------------------------------- |
| `MODEL`                 | `gpt-4o-mini-2024-07-18`                     | Model for API calls.                        |
| `API_URL`               | `https://api.openai.com/v1/chat/completions` | API endpoint.                               |
| `API_KEY_ENV`           | `API_KEY_PROJ`                               | Env var for API key.                        |
| `CONCURRENT_BATCH_SIZE` | 10                                           | Concurrent requests per batch.              |
| `CONFIDENCE_SCALE`      | 1                                            | Confidence scale: 1 = [0,1], 100 = [0,100]. |
| `OUTPUT_BASE`           | `popQA`                                      | Prefix for output filenames.                |
| `RESULTS_FOLDER`        | `preliminary_exp/outputs`                    | Output directory.                           |
| `NUM_QUESTIONS`         | `None`                                       | Limit questions (e.g. 10 for testing).      |


**Output format** (CSV):  
`dataset_id`, `timestamp`, `s_pop`, `o_pop`, `question`, `possible_answers`, `first_answer`, `first_confidence`, `best_guess`, `best_guess_confidence`, `correct`, `score`, `idk_flag`, `false_answer_flag`, `self_confidence`, `api_confidence_min`, `api_confidence_avg`

- **API confidence**: `api_confidence_min` = exp(min logprob), `api_confidence_avg` = geometric mean of logprobs over answer tokens (Answer or Best Guess).
- **IDK logic**: If the model says "I don't know", `self_confidence` and correctness use Best Guess values.

### Evaluation: `evaluate_PopQA.py`

**Input**: PopQA output CSV (set `input_file` in `__main__`).

**Metrics**:

- **ECE** (Expected Calibration Error) — self-confidence and API confidence (min, avg).
- **Brier score** — self-confidence and API confidence.
- **False answer rate** — 1 − accuracy.
- **Correlation** — Pearson, Spearman, Kendall (self vs API confidence).

---

## MMLU-Pro

### Script: `run_MMLU-pro.py`

**Parameters** (edit in `__main__`):


| Parameter        | Default                                      | Description                           |
| ---------------- | -------------------------------------------- | ------------------------------------- |
| `CATEGORY`       | `biology`                                    | Subject (e.g. biology).               |
| `PROMPT_FILE`    | `biology-1.yaml`                             | Prompt YAML in `prompts/{category}/`. |
| `OUTPUT_FILE`    | `biology-1`                                  | Output filename suffix.               |
| `OUTPUT_BASE`    | `MMLU-Pro`                                   | Prefix for output filenames.          |
| `API_URL`        | `https://api.openai.com/v1/chat/completions` | API endpoint.                         |
| `API_KEY_ENV`    | `API_KEY_ALT`                                | Env var for API key.                  |
| `RESULTS_FOLDER` | `preliminary_exp/outputs`                    | Output directory.                     |
| `BATCH_SIZE`     | 5                                            | Questions per batch.                  |
| `NUM_REPEATS`    | 10                                           | Repeats per question.                 |


**Output format** (CSV):  
`index`, `question`, `options`, `correct_answer`, `model_answers`, `model_answer`, `answer_distribution`, `self_confidences`, `api_confidences_key_token`

- **API confidence**: `api_confidences_key_token` = exp(logprob) of the chosen option token (e.g. "A", "B").
- **Prompt format**: No-CoT; last line is `"<option> <confidence>"` (e.g. `B 0.9500`).

### Evaluation: `evaluate_MMLU-Pro.py`

**Input**: MMLU-Pro output CSV (set `input_file` in `__main__`).

**Metrics**:

- **ECE** — median-aggregated per question (self-confidence, API key-token).
- **Brier score** — per-question average across repeats, then mean.
- **False answer rate** — question-level accuracy over repeats, then mean.
- **Correlation** — Pearson (self vs API key-token).

---

## Notes

1. **Model support**: Only models that return token-level log-probabilities in the API response are supported (e.g. `gpt-4o-mini`). For other models, change the model name and adjust the logprob extraction logic in the run scripts.
2. **Utils**: See `utils/README.md` for `api_caller` and `yaml_parser`.
3. **Environment**: Set the API key in `.env` or your environment (e.g. `API_KEY_PROJ`).

