import re
import json
import argparse
from modules.utils.read_utils import load_dataset

def read_pred_file(input):
    try:
        preds = {}
        with open(input, "r") as fin:
            for line in fin.readlines():
                preds.update(json.loads(line))
        return preds
    except:
        print(f"Error reading {input}")
        return None

def clean_answer(options, answer):
    for option, content in options.items():
        answer = answer.replace("Option", "")
        answer = answer.split(":")[0]
        if option in answer:
            return option
    return None

def match(gold_answer, answer):
    if isinstance(answer, list) and len(answer) > 0:
        answer = answer[0]
    if not isinstance(answer, str):
        return False
    pattern = r"\b[A-Z]\b|[A-Z](?=\s|:)"
    match = re.search(pattern, answer)
    if match is None:
        return False
    match = match.group()
    if match == gold_answer:
        return True
    return False

def compare(options, answer_1, answer_2):
    if type(answer_1) is list:
        answer_1 = answer_1[0]
    if type(answer_2) is list:
        answer_2 = answer_2[0]
    answer_1 = clean_answer(options, answer_1)
    answer_2 = clean_answer(options, answer_2)
    if answer_1 is None or answer_2 is None:
        return False
    if answer_1 == answer_2:
        return True
    return False
        

def main():
    parse = argparse.ArgumentParser()
    parse.add_argument("--data", type=str, default="caltech-101_image_heavy", help="the relative path of argments file")
    parse.add_argument("--input", type=str, default=None, help="the relative path of input file")
    parse.add_argument("--input_2", type=str, default=None, help="the relative path of input file")
    
    args = parse.parse_args()

    if "text_heavy" in args.data: ## Text heavy task
        mode = "text_heavy"
    elif "image_heavy" in args.data: ## Image heavy task
        mode = "image_heavy"
    elif "VQA" in args.data: ## Pure text task
        mode = "VQA"
    
    dataset_nickname, dataset = load_dataset(args.data, mode, 'test')

    target_answers = [chr(65 + i) for i in range(26)]
        
    preds = read_pred_file(args.input)
    cnt_correct = 0
    cnt = 0
    cnt_ignore = 0
    for data in dataset:
        try:
            data_id = data["id"]
        except:
            data_id = data["data_id"]
        pred = preds.get(data_id)
        if pred is None:
            cnt += 1
            continue
        options = data["multiple_choices"]
        gold_answer = data["multiple_choices_answer"]
        flag = match(gold_answer, pred)
        if not flag and pred not in target_answers:
            gold_answer_content = options[gold_answer]
            if options[gold_answer][0] == "'":
                gold_answer_content = options[gold_answer][1:]
            if options[gold_answer][-1] == "'":
                gold_answer_content = options[gold_answer][:-1]
            if gold_answer_content in pred:
                flag == True
            else:
                cnt += 1
                continue
        if flag:
            cnt_correct += 1
        cnt += 1
        cnt_ignore += 1
    print(f"Ignore Accuracy {cnt_correct} / {cnt_ignore}: {cnt_correct/cnt_ignore:.4f}")
    print(f"All Accuracy: {cnt_correct} / {cnt}: {cnt_correct/cnt:.4f}")
    print(f"{cnt_correct / cnt:.4f}({cnt_correct/cnt_ignore:.4f}, {cnt_correct}/{cnt_ignore})")

if __name__ == "__main__":
    main()   