import os
import re
import json
import argparse
from modules.utils.read_utils import load_dataset

def read_pred_file(input):
    preds = {}
    with open(input, "r") as fin:
        for line in fin.readlines():
            try:
                preds.update(json.loads(line))
            except:
                print(line)
                continue
    return preds

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

def safe_div(num, den):
    return float(num) / float(den) if den else 0.0

def record_result(store, dataset_nickname, sample, cnt_correct, cnt, cnt_ignore):
    if dataset_nickname not in store:
        store[dataset_nickname] = {}
    store[dataset_nickname][sample] = {
        "all_report_accuracy": round(safe_div(cnt_correct, cnt), 4),
        "ignore_accuracy": round(safe_div(cnt_correct, cnt_ignore), 4),
        "counts": {
            "correct": int(cnt_correct),
            "reported": int(cnt),
            "ignored": int(cnt) - int(cnt_ignore),
        },
    }

def main():
    parse = argparse.ArgumentParser()
    parse.add_argument("--data", type=str, default="caltech-101_image_heavy", help="the relative path of argments file")
    parse.add_argument("--model", type=str, default=None, help="the relative path of input file")
    parse.add_argument("--tag", type=str, default='', help="the relative path of input file")
    
    args = parse.parse_args()

    if "text_heavy" in args.data: ## Text heavy task
        mode = "text_heavy"
    elif "image_heavy" in args.data: ## Image heavy task
        mode = "image_heavy"
    elif "VQA" in args.data: ## Pure text task
        mode = "VQA"
    
    dataset_nickname, dataset = load_dataset(args.data, mode, 'test')

    target_answers = [chr(65 + i) for i in range(26)]
    
    result_store = {}
    print()
    print()
    if mode == "VQA":
        sample = 'origin'
        file_path = f"outputs/{dataset_nickname}/{args.model}/{mode}/{args.tag}/{dataset_nickname}_{mode}_origin_{args.tag}.txt"
        preds = read_pred_file(file_path)
        if preds:
            cnt_correct = 0
            cnt = 0
            cnt_ignore = 0
            for data in dataset:
                try:
                    data_id = data["id"]
                except:
                    data_id = data["data_id"]
                pred_info = preds.get(data_id)
                if pred_info:
                    try:
                        pred = pred_info.get("choice")
                        output = pred_info.get("answer")
                    except:
                        pred = pred_info
                else:
                    continue
                if not pred:
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
                    if output and gold_answer_content in output:
                        flag == True
                    else:
                        cnt += 1
                        continue
                if flag:
                    cnt_correct += 1
                cnt += 1
                cnt_ignore += 1
            print(f'{mode}, origin setting:')
            print(f"Ignore Accuracy {cnt_correct} / {cnt_ignore}: {safe_div(cnt_correct, cnt_ignore):.4f}")
            print(f"All Report Accuracy: {cnt_correct} / {cnt}: {safe_div(cnt_correct, cnt):.4f}")
            print(f"{safe_div(cnt_correct, cnt):.4f}({safe_div(cnt_correct, cnt_ignore):.4f}, {cnt_correct}/{cnt_ignore})")

            record_result(result_store, dataset_nickname, sample, cnt_correct, cnt, cnt_ignore)

    elif mode == "text_heavy":
        for sample in ['random', 'switch', 'full_black', 'full_white']:
            file_path = f"outputs/{dataset_nickname}/{args.model}/{mode}/{args.tag}/{dataset_nickname}_{mode}_{sample}_{args.tag}.txt"
            preds = read_pred_file(file_path)
            if preds:
                cnt_correct = 0
                cnt = 0
                cnt_ignore = 0
                for data in dataset:
                    try:
                        data_id = data["id"]
                    except:
                        data_id = data["data_id"]
                    pred_info = preds.get(data_id)
                    if pred_info:
                        try:
                            pred = pred_info.get("choice")
                            output = pred_info.get("answer")
                        except:
                            pred = pred_info
                    else:
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
                        if output and gold_answer_content in output:
                            flag == True
                        else:
                            cnt += 1
                            continue
                    if flag:
                        cnt_correct += 1
                    cnt += 1
                    cnt_ignore += 1
                print(f'{mode}, {sample} setting:')
                print(f"Ignore Accuracy {cnt_correct} / {cnt_ignore}: {safe_div(cnt_correct, cnt_ignore):.4f}")
                print(f"All Report Accuracy: {cnt_correct} / {cnt}: {safe_div(cnt_correct, cnt):.4f}")
                print(f"{safe_div(cnt_correct, cnt):.4f}({safe_div(cnt_correct, cnt_ignore):.4f}, {cnt_correct}/{cnt_ignore})")

                record_result(result_store, dataset_nickname, sample, cnt_correct, cnt, cnt_ignore)

    elif mode == "image_heavy":
        for sample in ['origin', 'unrelated_text', 'related_text']:
            file_path = f"outputs/{dataset_nickname}/{args.model}/{mode}/{args.tag}/{dataset_nickname}_{mode}_{sample}_{args.tag}.txt"
            preds = read_pred_file(file_path)
            if preds:
                cnt_correct = 0
                cnt = 0
                cnt_ignore = 0
                for data in dataset:
                    try:
                        data_id = data["id"]
                    except:
                        data_id = data["data_id"]
                    pred_info = preds.get(data_id)
                    if pred_info:
                        try:
                            pred = pred_info.get("choice")
                            output = pred_info.get("answer")
                        except:
                            pred = pred_info
                    else:
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
                        if output and gold_answer_content in output:
                            flag == True
                        else:
                            cnt += 1
                            continue
                    if flag:
                        cnt_correct += 1
                    cnt += 1
                    cnt_ignore += 1
                print(f'{mode}, {sample} setting:')
                print(f"Ignore Accuracy {cnt_correct} / {cnt_ignore}: {safe_div(cnt_correct, cnt_ignore):.4f}")
                print(f"All Report Accuracy: {cnt_correct} / {cnt}: {safe_div(cnt_correct, cnt):.4f}")
                print(f"{safe_div(cnt_correct, cnt):.4f}({safe_div(cnt_correct, cnt_ignore):.4f}, {cnt_correct}/{cnt_ignore})")

                record_result(result_store, dataset_nickname, sample, cnt_correct, cnt, cnt_ignore)
    else:
        raise("Invalid mode")
    
    print()

    save_dir = os.path.join("results", dataset_nickname, str(args.model), str(args.tag))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "result.txt")
    with open(save_path, "w") as f:
        json.dump(result_store, f, indent=2)
    print(f"[Saved] metrics json -> {save_path}")


if __name__ == "__main__":
    main()   