import argparse
from tkinter import NONE
import pandas as pd
from typing import Any
from math_verify.metric import math_metric
from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig, parse
import sympy
from pathlib import Path
from typing import Iterable, Union
import json
import os
def parse_args():
    parser = argparse.ArgumentParser(description='Extract and evaluate answers using sympy')
    parser.add_argument('--input_jsonl', type=str, required=True, help='Path to input josnl file containing model outputs')
    parser.add_argument('--output_jsonl', type=str, required=True, help='Path to output jsonl file for extracted answers')
    parser.add_argument('--gold_is_latex', action='store_true', help='Use basic latex normalization', default=True)
    return parser.parse_args()

def load_csv_data(csv_path: str) -> pd.DataFrame:
    """Load and validate CSV data."""
    try:
        df = pd.read_csv(csv_path)
        required_columns = ['answer', 'gold']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")
        return df
    except Exception as e:
        raise Exception(f"Error loading CSV file: {str(e)}")


def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                print("Error in loading:", line)
                exit()


def save_jsonl(samples, save_path):
    # ensure path
    folder = os.path.dirname(save_path)
    os.makedirs(folder, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print("Saved to", save_path)


def serialize_sympy_object(obj: Any) -> str:
    """Convert sympy object to string representation."""
    if obj is None:
        return ""
    try:
        if isinstance(obj, (list, tuple)):
            return ", ".join(str(x) if x is not None else "" for x in obj)
        return str(obj)
    except Exception as e:
        return f"Error: {str(e)}"

def compare_answers(extracted: Any, gold: Any) -> bool:
    """Compare extracted answer with gold answer."""
    if extracted is None or gold is None:
        return False
    try:
        # Handle lists/tuples of expressions
        if isinstance(extracted, (list, tuple)) and isinstance(gold, (list, tuple)):
            if len(extracted) != len(gold):
                return False
            return all(sympy.simplify(a - b) == 0 for a, b in zip(extracted, gold))
        
        # Handle single expressions
        return sympy.simplify(extracted - gold) == 0
    except Exception:
        # If comparison fails (e.g. different types), return False
        return False

def process_answers(input_list: list[dict], gold_is_latex: bool) -> Union[list[dict], dict]:
    """Process each answer through the sympy extraction workflow and compare with gold using math_verify."""
    results = []
    
    code_len = None

    correct_count = None
    any_correct_count = 0
    total_count = 0
    
    # Create the verification function
    verify_func = math_metric(
        # gold_extraction_target=(LatexExtractionConfig() if gold_is_latex else ExprExtractionConfig(),),
        gold_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        aggregation_function=max,
        precision=6
    )
    output_list = []
    for item in input_list:
        if code_len is None:
            code_len = len(item.get('code', []))
            correct_count = [0 for _ in range(code_len)]

        # Evaluate all responses in the code list
        code_responses = item.get('code', [])
        if not isinstance(code_responses, list):
            code_responses = [code_responses]
        
        item_results = []
        item_correct = False
        
        for i, code_response in enumerate(code_responses):
            extracted_answers = None
            gold_answers = None
            grade = 0
            try:
                # Use the verification function
                grade, extracted_answers = verify_func([item['answer']], [code_response])
                if extracted_answers is None:
                    extracted_answers = None
                    gold_answers = None
                else:
                    gold_answers = extracted_answers[0]
                    extracted_answers = extracted_answers[1]
                
                if grade != 1:
                    grade, extracted_answers = verify_func([code_response], [item['answer']])
                if extracted_answers is None:
                    extracted_answers = None
                    gold_answers = None
                else:
                    gold_answers = extracted_answers[1]
                    extracted_answers = extracted_answers[0]

                is_correct = grade == 1
                if is_correct:
                    item_correct = True
                
                response_result = {
                    'response_index': i,
                    'extracted_answer': extracted_answers,
                    'is_correct': is_correct,
                }
                item_results.append(response_result)
                
            except Exception as e:
                response_result = {
                    'response_index': i,
                    'extracted_answer': extracted_answers,
                    'is_correct': False,
                    'error': str(e)
                }
                item_results.append(response_result)
        
        total_count += 1
        for i in range(code_len):
            if item_results[i]['is_correct']:
                correct_count[i] += 1
        
        # Add evaluation results to the item
        item['code_evaluations'] = item_results
        item['any_correct'] = item_correct
        any_correct_count += item_correct
        item['num_responses'] = len(code_responses)
        item['num_correct'] = sum(1 for r in item_results if r['is_correct'])
        output_list.append(item)

    # Calculate accuracy
    accuracy = []
    for i in range(code_len):
        accuracy.append(correct_count[i] / total_count if total_count > 0 else 0)
    print(f"\nEvaluation Results:")
    print(f"Total examples: {total_count}")
    print(f"Correct answers: {correct_count}")
    print(f"Accuracy: {accuracy}")
    print(f"Pass@k: {any_correct_count / total_count if total_count > 0 else 0}")
    # Calculate mean and std of accuracy
    accuracy_mean = sum(accuracy) / len(accuracy) if accuracy else 0
    accuracy_std = (sum((x - accuracy_mean) ** 2 for x in accuracy) / len(accuracy)) ** 0.5 if accuracy else 0
    print(f"Accuracy mean: {accuracy_mean:.4f}")
    print(f"Accuracy std: {accuracy_std:.4f}")
    
    # Add summary stats to the dataframe
    result = {
        'accuracy': accuracy,
        'pass@k': any_correct_count / total_count if total_count > 0 else 0,
        'total_count': total_count,
        'correct_count': correct_count
    }
    return output_list, result

def main():
    args = parse_args()
    
    # Load input CSV
    input_list = load_jsonl(args.input_jsonl)
    
    # Process answers and extract sympy objects
    output_list, result_json = process_answers(input_list, args.gold_is_latex)
    
    # Save results to output CSV
    # results_df.to_csv(args.output_csv, index=False)
    save_jsonl(output_list, args.output_jsonl)
    with open(
        args.output_jsonl.replace(".jsonl", f"_metrics.json"), "w"
    ) as f:
        json.dump(result_json, f, indent=4)
    print(f"\nResults saved to {args.output_jsonl}")

if __name__ == "__main__":
    main()


