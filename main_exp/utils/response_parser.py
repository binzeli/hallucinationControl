"""
Utility functions for parsing model responses.
"""

import re


def extract_fields(text):
    """
    Parse single-response format; handles blank confidence and % signs.
    
    Args:
        text: The response text to parse
        
    Returns:
        Tuple of (answer, confidence, best_guess, best_guess_confidence)
    """
    text = text.strip().replace("\r\n", "\n")

    ans_match = re.search(r"(?im)^\s*(?:answer|response)\s*:\s*(.+?)\s*$", text)
    conf_match = re.search(r"(?im)^\s*confidence\s*:\s*([0-9]+(?:\.[0-9]+)?)?\s*%?\s*$", text)
    best_match = re.search(r"(?im)^\s*best\s*guess\s*:\s*(.+?)\s*$", text)
    best_conf_match = re.search(r"(?im)^\s*best\s*guess\s*confidence\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%?\s*$", text)

    answer = ans_match.group(1).strip() if ans_match else None
    conf = float(conf_match.group(1)) if (conf_match and conf_match.group(1)) else None
    best_guess = best_match.group(1).strip().rstrip(",") if best_match else None
    if best_guess and best_guess.lower() in {"", "none", "n/a"}:
        best_guess = None
    best_conf = float(best_conf_match.group(1)) if best_conf_match else None

    return answer, conf, best_guess, best_conf


def extract_fields_gsm8k(text):
    """
    Parse response format for GSM8K dataset.
    Extracts answer and confidence similar to extract_fields but adapted for math problems.
    
    Args:
        text: The response text to parse
        
    Returns:
        Tuple of (answer, confidence, best_guess, best_guess_confidence)
    """
    text = text.strip().replace("\r\n", "\n")

    ans_match = re.search(r"(?im)^\s*(?:answer|response)\s*:\s*(.+?)\s*$", text)
    conf_match = re.search(r"(?im)^\s*confidence\s*:\s*([0-9]+(?:\.[0-9]+)?)?\s*%?\s*$", text)
    best_match = re.search(r"(?im)^\s*best\s*guess\s*:\s*(.+?)\s*$", text)
    best_conf_match = re.search(r"(?im)^\s*best\s*guess\s*confidence\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%?\s*$", text)

    answer = ans_match.group(1).strip() if ans_match else None
    conf = float(conf_match.group(1)) if (conf_match and conf_match.group(1)) else None
    best_guess = best_match.group(1).strip().rstrip(",") if best_match else None
    if best_guess and best_guess.lower() in {"", "none", "n/a"}:
        best_guess = None
    best_conf = float(best_conf_match.group(1)) if best_conf_match else None

    return answer, conf, best_guess, best_conf
    