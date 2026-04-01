# hallucinationControl
I-CALM: Incentivizing Confidence-Aware Abstention for LLM Hallucination Mitigation

## Overview

LLMs frequently produce confident but incorrect answers, partly because standard scoring conventions reward answering over expressing uncertainty. We study whether prompt-only interventions—announcing explicit reward schemes for answer-versus-abstain decisions alongside truthfulness- and humility-oriented norms—can reduce hallucination risk without modifying the model.

We introduce **I-CALM**, a prompt-based framework that (i) elicits verbal confidence, (ii) partially rewards abstention via explicit reward schemes, and (iii) adds lightweight truthfulness- and humility-oriented norms. Experiments on PopQA show that abstention-rewarding prompts, especially with norms, lower the false-answer rate by shifting low-confidence cases into abstention, with a clear abstention–hallucination trade-off controlled by the abstention reward. Results demonstrate that selective answering on factual questions can be improved without retraining, though the effect size varies across models and datasets.

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd hallucinationControl
```

### 2. Create and activate virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```bash
API_KEY=your_openai_api_key_here
```

## Experiment Schemes

Each scheme varies what reward scheme is disclosed to the model and whether a normative system prompt is added:

| Scheme | Description | Rewards Mentioned | System Prompt |
|--------|-------------|-------------------|---------------|
| **pure_eval** | Pure evaluation with no reward information | None | No |
| **scheme_a_baseline** | Simplified baseline with rewards but no IDK mention | Correct, Incorrect | No |
| **scheme_a** | Rewards excluding IDK from prompt text | Correct, Incorrect | No |
| **scheme_b** | Full rewards including IDK incentive | Correct, Incorrect, Abstain | No |
| **scheme_b_norm** | Same as scheme_b with normative system prompt | Correct, Incorrect, Abstain | Yes |

## Usage

### Available Models

- `gpt-4o-mini` - GPT-4o Mini 
- `gpt-5-mini` - GPT-5 Mini
- `qwen-3-4b` - Qwen3-4B-Instruct-2507
- `gemini-3.1-flash-lite` - Gemini 3.1 Flash Lite
- `llama-3-8b` - Meta-Llama-3-8B-Instruct


### Command-Line Arguments

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--model` | `-m` | str | required | Model to use |
| `--scheme` | `-s` | str | scheme_b | Experiment scheme |
| `--samples` | `-n` | int | None | Number of samples (None = full dataset) |
| `--reward-correct` | `-rc` | float | 0.0 | Reward for correct answer |
| `--reward-abstain` | `-ra` | float | 0.0 | Reward for abstaining |
| `--reward-incorrect` | `-ri` | float | 0.0 | Penalty for incorrect answer |

### Examples

```bash
python main_exp/popQA/run_experiment.py --model gpt-5-mini --scheme scheme_b_norm --samples 1000 -rc 1 -ra 0.4 -ri -1
python3 main_exp/popQA/run_experiment.py --model gpt-4o-mini --scheme scheme_a --samples 10 -rc 1 -ri -1
```

## Evaluation

**Main experiments (PopQA / MMLU-Pro):** analyze a result CSV and generate plots:

```bash
python main_exp/popQA/eval.py -f <path_to_csv>
python main_exp/mmlu-pro/eval.py -f <path_to_csv>
```

**Preliminary experiments:** run evaluation (ECE, Brier, FAR, correlation) by passing the result file:

```bash
python preliminary_exp/evaluate_PopQA.py -f <path_to_popqa_csv>
python preliminary_exp/evaluate_MMLU-Pro.py -f <path_to_mmlu_csv>
```

Optional: `-o <output_dir>` for outputs, `--num-bins <n>` for ECE (default 10).



## Project Structure

```
├── main_exp/                   # Main (reward-scheme) experiments
│   ├── popQA/                  # PopQA experiments, plots, ablation studies
│   ├── mmlu-pro/               # MMLU-Pro experiments (standard & CoT)
│   ├── simpleQA-verified/      # SimpleQA-verified experiments
│   ├── TriviaQA/               # TriviaQA experiments
│   ├── prompts/                # Shared prompt templates
│   ├── utils/                  # Response & abstention parsers
│   └── example_output/         # Example result CSVs & plots
├── preliminary_exp/            # Preliminary: validity of self-reported confidence
│   ├── run_PopQA.py            # PopQA runner (logprobs, IDK/best-guess)
│   ├── run_MMLU-pro.py         # MMLU-Pro runner
│   ├── evaluate_PopQA.py       # ECE, Brier, correlation metrics
│   ├── evaluate_MMLU-Pro.py    # Same metrics for MMLU-Pro
│   ├── utils/                  # API caller & YAML parser
│   └── example_output/         # Example result CSVs
├── requirements.txt
├── .env                        # API keys (create this file)
└── README.md
```
