"""
Parser for scheme_a_baseline results to detect uncertainty phrases and set idk_flag.
Since scheme_a_baseline doesn't explicitly ask for "I don't know", we need to detect
uncertainty phrases in the model's responses.
"""

import pandas as pd
import re
import os
import argparse


# Define uncertainty phrases to check for
UNCERTAINTY_PHRASES = [
    "not certain",
    "not sure",
    "I don't have",
    "t have enough",
    "don't know",
    "do not know",
    "Could you",
    ". which",
    "unknown"
]


def check_uncertainty(answer):
    """
    Check if any uncertainty phrase is in the answer.
    
    Args:
        answer: The answer text to check
        
    Returns:
        1 if uncertainty detected, 0 otherwise
    """
    if pd.isna(answer):
        return 0
    
    answer_str = str(answer).lower()
    
    # Check for each uncertainty phrase (case-insensitive)
    for phrase in UNCERTAINTY_PHRASES:
        if phrase.lower() in answer_str:
            return 1
    
    return 0


def parse_csv(input_file, output_file=None):
    """
    Parse scheme_a_baseline CSV results and add idk_flag based on uncertainty detection.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to save the output (if None, overwrites input_file)
        
    Returns:
        DataFrame with added idk_flag column
    """
    # Read the CSV file
    print(f"📖 Reading file: {input_file}")
    df = pd.read_csv(input_file)
    
    # Add the idk_flag column
    print("🔍 Detecting uncertainty phrases...")
    df['idk_flag'] = df['first_answer'].apply(check_uncertainty)
    
    # Recalculate scores based on updated idk_flag
    # For baseline: +1 for correct, 0 for incorrect or IDK
    print("📊 Recalculating scores...")
    
    def recalculate_score(row):
        if row['idk_flag'] == 1:
            # If IDK was detected, check if best_guess is correct
            # But for baseline, there's no best_guess, so we treat it as abstain (score = 0)
            return 0
        elif row['correct'] == 1:
            return 1
        else:
            return 0
    
    df['score'] = df.apply(recalculate_score, axis=1)
    
    # Update false_answer_flag: 1 if not correct and not IDK
    df['false_answer_flag'] = ((df['correct'] == 0) & (df['idk_flag'] == 0)).astype(int)
    
    # Save the modified dataframe
    if output_file is None:
        output_file = input_file
    
    df.to_csv(output_file, index=False)
    
    print(f"\n✅ Processed file: {input_file}")
    print(f"💾 Output saved to: {output_file}")
    print(f"\n📈 Statistics:")
    print(f"Total rows: {len(df)}")
    print(f"Rows with uncertainty (idk_flag=1): {df['idk_flag'].sum()}")
    print(f"Percentage with uncertainty: {df['idk_flag'].sum() / len(df) * 100:.2f}%")
    print(f"Rows without uncertainty (idk_flag=0): {len(df) - df['idk_flag'].sum()}")
    print(f"Percentage without uncertainty: {(len(df) - df['idk_flag'].sum()) / len(df) * 100:.2f}%")
    print(f"\nCorrect answers: {df['correct'].sum()} ({df['correct'].sum() / len(df) * 100:.2f}%)")
    print(f"False answers: {df['false_answer_flag'].sum()} ({df['false_answer_flag'].sum() / len(df) * 100:.2f}%)")
    print(f"Average score: {df['score'].mean():.4f}")
    
    return df

