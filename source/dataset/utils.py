'''Dataset utilities.'''

import json
import re
import csv
from collections import Counter
from fractions import Fraction
import math
import string
from typing import List, Tuple, Dict, Any, Union

INVALID_ANS = "[invalid]" 

ERROR_ANS = "[error]"

MAX_TOKEN = {
    "DU": 1000,
    "WS": 1000,
    "MA": 1000,
    "StrategyQA": 1000,
    "LLC": 1000,
    "MATH": 2000,
}

TASK_DEFINITIONS = {
	"DU": "The task is to solve the date understanding problem.",
	"WS": "The task is to sort a list of words.",
	"MA": "The task is to solve multi-step arithmetic problems.",
	"StrategyQA": "The task requires models to infer a multi-hop strategy to answer questions.",
	"LLC": "The task is to solve the last letter concatenation problem.",
}


def post_process(answer: str, dataset_name: str) -> str:
    """Post-process the answer.

    Args:
        answer (str): The answer.
        dataset_name (str): The name of the dataset.
    
    Returns:
        str: The post-processed answer.
    """
    if dataset_name in ["GSM8K", "MATH"]:
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()
              
        if "is" in answer:
            answer = answer.split("is")[-1].strip()

        if "as" in answer:
            answer = answer.split("as")[-1].strip()

        if "to" in answer:
            answer = answer.split("to")[-1].strip()

        if "=" in answer:
            answer = answer.split("=")[-1].strip()

    return answer


def is_correct(pred_answer: Any, gold_answer: Any) -> bool:
    """Check if the predicted answer is correct.
    
    Args:
        pred_answer (Any): The predicted answer.
        gold_answer (Any): The gold answer.
        
    Returns:
        bool: Whether the predicted answer is correct.
    """

    return pred_answer == gold_answer


def evaluate_acc(
        dataset: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
        dataset_name: str,
        non_empty_only: bool = False,
        valid_only: bool = False,
        debug: bool = False) -> Tuple[float, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Evaluate the accuracy of the predictions.
    
    Args:
        dataset (list): The dataset.
        predictions (list): The predictions.
        dataset_name (str): The name of the dataset.
        non_empty_only (bool): Whether to only consider non-empty predictions.
        valid_only (bool): Whether to only consider valid predictions.
        debug (bool): Whether to only consider the first 10 examples.
    
    Returns:
        Tuple[float, list, list]: The accuracy, the correct examples, and the incorrect examples.
    """
    correct_count, total_count = 0, 0
    correct_examples = []
    incorrect_examples = []
    for i, (example, prediction) in enumerate(zip(dataset, predictions)):
        gold_id = int(example["id"]) if "id" in example else i
        if "id" not in example:
            example["id"] = i

        if prediction == {}:
            continue
        pred_id = int(prediction["id"])

        try:
            assert gold_id == pred_id
        except:
            raise AssertionError(f"Gold id {gold_id} doesn't match pred id {pred_id}.")

        try:
            gold_answer = extract_gold_answer(dataset_name, example["answer"])
        except SyntaxError as e:
            print("Error: ", e)
            print(gold_id)
            exit(-1)
        pred_answer = extract_pred_answer(dataset_name, post_process(prediction["answer"], dataset_name))

        if non_empty_only and pred_answer == "":
            continue

        if valid_only:
            if type(pred_answer) == str and ("invalid" in pred_answer or "error" in pred_answer):
                continue

        total_count += 1

        try:
            correct = is_correct(pred_answer, gold_answer)
        except Exception as e:
            exit(-1)

        if correct:
            correct_count += 1
            correct_examples.append(example)
        else:
            incorrect_examples.append(example)

        if debug and total_count >= 10:
            break

    acc = round(correct_count / total_count * 100, 1)

    return acc, correct_examples, incorrect_examples


def _fix_fracs(string: str) -> str:
    """Fix fractions in the string.
    
    Args:
        string (str): The string.
        
    Returns:
        str: The string with fixed fractions.
    """
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string: str) -> str:
    """Fix a/b in the string. e.g. 1/2 --> \frac{1}{2}
	
	Args:
        string (str): The string.
	
	Returns:
        str: The string with fixed fractions.
    """
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def _remove_right_units(string: str) -> str:
    """Remove units on the right side of the string. e.g. 5\text{cm} --> 5
	
	Args:
        string (str): The string.
		
	Returns:
        str: The string with units removed.
    """
    if "\\text{" in string:
        splits = string.split("\\text{")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def _fix_sqrt(string: str) -> str:
    """Fix sqrt in the string. e.g. sqrt3 --> sqrt{3}

    Args:
        string (str): The string.
    
    Returns:
        str: The string with fixed sqrt.
    """
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string: str) -> str:
    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("$.", "$")
    string = string.replace("\\$", "")
    string = string.replace("$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace("}.", "}")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1).
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset
    string = _fix_a_slash_b(string)

    return string


def normalize(s: str) -> str:
    """Normalize a string. Remove punctuation and extra white spaces and convert to lower case.

    Args:
        s (str): The string.
    
    Returns:
        str: The normalized string.
    """
    def white_space_fix(s):
        return ' '.join(s.split())
    def remove_punc(s):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in s if ch not in exclude)
    def lower(s):
        return s.lower()

    return lower(white_space_fix(remove_punc(s)))


def extract_answer(pred_str: str, dataset_name: str) -> str:
    """Extract the answer from a completion.

    Args:
        pred_str (str): The completion.
        dataset_name (str): The name of the dataset.
    
    Returns:
        str: The extracted answer.
    """
    if dataset_name in ["LLC", "WS", "StrategyQA", "DU"]:
        if 'The answer is ' in pred_str:
            pred = pred_str.split('The answer is ')[-1].strip()
        elif 'the answer is ' in pred_str:
            pred = pred_str.split('the answer is ')[-1].strip()
        elif 'The final answer is ' in pred_str:
            pred = pred_str.split('The final answer is ')[-1].strip()
        elif 'the final answer is ' in pred_str:
            pred = pred_str.split('the final answer is ')[-1].strip()
        else:
            pred = pred_str.split()[-1].strip()
        return normalize(pred)
    else:
        return extract_math_answer(pred_str)
    

def extract_math_answer(pred_str: str) -> str:
    """Extract the math answer from a completion. This function is used for math datasets.

    Args:
        pred_str (str): The completion.
    
    Returns:
        str: The extracted math answer.
    """
    if 'The answer is ' in pred_str:
        pred = pred_str.split('The answer is ')[-1].strip()
    elif 'the answer is ' in pred_str:
        pred = pred_str.split('the answer is ')[-1].strip()
    elif 'The final answer is ' in pred_str:
        pred = pred_str.split('The final answer is ')[-1].strip()
    elif 'the final answer is ' in pred_str:
        pred = pred_str.split('the final answer is ')[-1].strip()
    elif "Answer:" in pred_str and " is " in pred_str.split('Answer:')[-1].strip():
        pred_str = pred_str.split('Answer:')[-1].strip()
        pred_str = pred_str.split(" is ")[-1].strip()
        if "$" in pred_str:
            pattern = r"\$(.*?)\$"
            pred = re.findall(pattern, pred_str)
            if len(pred) >= 1:
                pred = pred[-1]
            else:
                pattern = '-?\d*\.?\d+'
                pred = re.findall(pattern, pred_str)
                if len(pred) >= 1:
                    pred = pred[-1]
                else:
                    pred = ''
        else:
            pattern = '-?\d*\.?\d+'
            pred = re.findall(pattern, pred_str)
            if len(pred) >= 1:
                pred = pred[-1]
            else:
                pred = ''
    elif 'boxed' in pred_str:
        ans = pred_str.split('boxed')[-1]
        if ans[0] == '{':
            stack = 1
            a = ''
            for c in ans[1:]:
                if c == '{':
                    stack += 1
                    a += c
                elif c == '}':
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split('$')[0].strip()
        pred = a
    else:
        pattern = '-?\d*\.?\d+'
        pred = re.findall(pattern, pred_str)
        if len(pred) >= 1:
            pred = pred[-1]
        else:
            pred = ''
    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]
        if pred[-1] == "/":
            pred = pred[:-1]
    pred = _strip_string(pred)
    if 'boxed' in pred:
        ans = pred.split('boxed')[-1]
        if ans[0] == '{':
            stack = 1
            a = ''
            for c in ans[1:]:
                if c == '{':
                    stack += 1
                    a += c
                elif c == '}':
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split('$')[0].strip()
        a = _strip_string(a)
        pred = a
    return pred


def load_data(frn: str) -> List[Dict[str, Any]]:
    """Load data from a file.

    Args:
        frn (str): The file name.
    
    Returns:
        list: The data.
    """
    if frn.endswith(".jsonl"):
        with open(frn, 'r') as fr:
            lines = []
            for i, line in enumerate(fr):
                if line.strip() == "":
                    continue
                try:
                    lines.append(json.loads(line))
                except json.decoder.JSONDecodeError as e:
                    print(f"Error in line {i}: {line}\n {e}")
                    exit(-1)
        return lines
    elif frn.endswith(".csv"):
        with open(frn) as fr:
            reader = csv.DictReader(fr)
            return [line for line in reader]
    elif frn.endswith(".json"):
        with open(frn) as fr:
            data = json.load(fr)
            if type(data) is dict:
                return data["examples"]
            else:
                return data

def str2num(answer_str, rounding: str = "int", abs_val: bool = False) -> Union[int, float]:
    '''Convert a string to a number.

    Args:
        answer_str (str): The string.
        rounding (str): The rounding method for the answer. Can be "int", "ceil", or "floor".

    Returns:
        float: The number.
    '''
    if "/" in answer_str:
        answer_str = float(sum(Fraction(s) for s in answer_str.split()))
    answer_str = float(answer_str)

    if rounding == "int":
        answer_str = int(answer_str)
    elif rounding == "ceil":
        answer_str = math.ceil(answer_str)
    elif rounding == "floor":
        answer_str = math.floor(answer_str)

    if abs_val:
        answer_str = abs(answer_str)

    return answer_str


def extract_gold_answer(dataset_name: str, gold_completion: str) -> Any:
    '''Extract the gold answer from a completion.

    Args:
        dataset_name (str): The name of the dataset.
        gold_completion (str): The gold completion.
    
    Returns:
        Any: The gold answer.
    '''
    
    if dataset_name in ["GSM8K", "MA"]:
        gold_completion = gold_completion.replace(",", "")
        return int(gold_completion)
    elif dataset_name in ["DU"]:
        answer = gold_completion.split("#### ")[-1]
        return answer
    elif dataset_name in ["StrategyQA"]:
        answer = bool(gold_completion)
        return answer
    elif dataset_name in ["WS"]:
        answer = normalize(gold_completion)
        return answer
    else:
        return gold_completion


def extract_pred_answer(dataset_name: str, pred_completion: str, rounding: str = "int") -> Any:
    '''Extract the predicted answer from a completion.

    Args:
        dataset_name (str): The name of the dataset.
        pred_completion (str): The predicted completion.
        rounding (str): The rounding method for the answer. Can be "int", "ceil", or "floor".
    
    Returns:
        Any: The predicted answer.
    '''
    if INVALID_ANS in str(pred_completion):
        return INVALID_ANS

    if ERROR_ANS in str(pred_completion):
        return ERROR_ANS

    if dataset_name in ["GSM8K", "MA"]:
        if type(pred_completion) in [int, float]:
            pred_answer = pred_completion
        else:
            assert type(pred_completion) == str
            ANS_RE = re.compile(r"(\-?[0-9\.\,]+)")
            match = ANS_RE.search(pred_completion)
            if match:
                match_str = match.group(1).strip()
                match_str = match_str.replace(",", "")
                try:
                    abs_val = True if dataset_name in ["GSM8K"] else False
                    pred_answer = str2num(match_str, rounding, abs_val)
                except:
                    pred_answer = INVALID_ANS
            else:
                pred_answer = INVALID_ANS
        return pred_answer
    elif dataset_name in ["MATH"]:
        extract_number = True
        for x in ["\\frac", "\\pi", "\\sqrt", "\\text", "\\infty", "\\cup", "\\cot", "\\mbox", "-", "+", "^", "(", ")", "[", "]", "<", ">"]:
            if x in pred_completion:
                extract_number = False
        if extract_number:
            pattern = '-?\d*\.?\d+'
            pred = re.findall(pattern, pred_completion)
            if len(pred) >= 1:
                pred = pred[-1]
            else:
                pred = pred_completion
            return pred
        else:
            return pred_completion
    elif dataset_name in ["StrategyQA"]:
        answer = bool(pred_completion)
        return answer
    else:
        return pred_completion
