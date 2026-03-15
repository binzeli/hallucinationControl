# hallucinationControl
Incentivizing Confidence-Aware Abstention for LLM Hallucination Control

## Overview

This project explores methods to control hallucinations in Large Language Models (LLMs) by incentivizing confidence-aware abstention. The system tests whether reward-based prompting can encourage models to say "I don't know" when uncertain, rather than providing incorrect answers.

## Project Structure

```
├── main_exp/                           # Main (reward-scheme) experiments
│   ├── models/
│   │   └── gpt_client.py               # Generic GPT batch API client
│   ├── popQA/
│   │   ├── run_experiment.py           # Main entry for PopQA reward experiments
│   │   ├── eval.py                     # Evaluation for main_exp PopQA
│   │   ├── plot/
│   │   │   ├── plot_abstention.py
│   │   │   └── far_vs_abstention.py
│   │   └── ablation_study/
│   │       ├── remove_reward/run_no_reward.py
│   │       └── remove_confidence/run_no_confidence.py
│   ├── prompts/
│   │   └── prompts.py                  # Experiment prompt templates
│   ├── utils/
│   │   ├── response_parser.py          # Parse model responses
│   │   └── abstain_parser.py           # Detect abstention patterns
│   ├── rare_common_facts.py
│   └── example_output/                 # Example run outputs
├── preliminary_exp/                     # Validity of self-reported confidence
│   ├── README.md                       # PopQA & MMLU-Pro pipeline docs
│   ├── run_PopQA.py                    # PopQA API runner (logprobs, IDK/best-guess)
│   ├── run_MMLU-pro.py                 # MMLU-Pro API runner (no-CoT, repeats)
│   ├── evaluate_PopQA.py               # ECE, Brier, correlation (numerical only)
│   ├── evaluate_MMLU-Pro.py            # Same metrics for MMLU-Pro CSVs
│   ├── utils/
│   │   ├── README.md
│   │   ├── api_caller.py               # Concurrent API calls, logprobs
│   │   └── yaml_parser.py              # YAML load/save
│   ├── prompts/
│   │   └── biology/
│   │       └── biology-1.yaml          # No-CoT prompt template
│   ├── example_output/                 # Example result CSVs
│   └── outputs/                        # Result CSVs (create dir; see preliminary_exp/README)
├── requirements.txt
├── .env                                # API keys (create this file)
├── .gitignore
└── README.md
```

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

The project implements five experimental schemes:

| Scheme | Description | Rewards Mentioned | System Prompt |
|--------|-------------|-------------------|---------------|
| **pure_eval** | Pure evaluation with no reward information | None | No |
| **scheme_a_baseline** | Simplified baseline with rewards but no IDK mention | Correct, Incorrect | No |
| **scheme_a** | Rewards excluding IDK from prompt text | Correct, Incorrect, Abstain | No |
| **scheme_b** | Full rewards including IDK incentive | Correct, Incorrect, Abstain | No |
| **scheme_b_norm** | Same as scheme_b with normative system prompt | Correct, Incorrect, Abstain | Yes |

### Default Reward Values

- **Correct answer**: +1.0
- **Incorrect answer**: -1.0
- **Abstain ("I don't know")**: +0.4

## Usage

### Basic Command

```bash
python main_exp/popQA/run_experiment.py --model <model_name> --scheme <scheme_name>
```

### Available Models

- `gpt-4o-mini` - GPT-4o Mini (no reasoning mode)
- `gpt-5-mini` - GPT-5 Mini (with reasoning mode)

### Examples

Run scheme_b_norm with GPT-5:
```bash
python main_exp/popQA/run_experiment.py --model gpt-5-mini --scheme scheme_b_norm --samples 1000 -rc 1 -ra 0.4 -ri -1
python3 main_exp/popQA/run_experiment.py --model gpt-4o-mini --scheme scheme_a --samples 10 -rc 1 -ri -1
```

### Command-Line Arguments

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--model` | `-m` | str | required | Model to use (gpt-4o-mini, gpt-5-mini) |
| `--scheme` | `-s` | str | scheme_b | Experiment scheme |
| `--samples` | `-n` | int | None | Number of samples (None = full dataset) |
| `--reward-correct` | `-rc` | float | 1.0 | Reward for correct answer |
| `--reward-abstain` | `-ra` | float | 0.4 | Reward for abstaining |
| `--reward-incorrect` | `-ri` | float | -1.0 | Penalty for incorrect answer |

## Evaluation

After running experiments, use `eval.py` to analyze results and generate visualizations:

```bash
python main_exp/popQA/eval.py --result-file <path_to_csv>
```
