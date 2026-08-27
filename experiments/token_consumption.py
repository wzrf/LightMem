import json
from collections import defaultdict
from pathlib import Path
import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import numpy as np


def simple_tokenize(text):
    """简单的分词与文本清洗函数（用于计算 F1）"""
    text = str(text)
    return (
        text.lower()
        .replace(".", " ")
        .replace(",", " ")
        .replace("!", " ")
        .replace("?", " ")
        .split()
    )


def compute_f1(prediction, reference):
    """基于词重叠（Overlap Token Level）计算 F1 Score"""
    pred_tokens = set(simple_tokenize(prediction))
    ref_tokens = set(simple_tokenize(reference))
    common_tokens = pred_tokens & ref_tokens

    if not pred_tokens or not ref_tokens:
        return 0.0

    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(ref_tokens)

    if (precision + recall) > 0:
        return 2 * precision * recall / (precision + recall)
    return 0.0


def calculate_bleu_scores(prediction: str, reference: str):
    """使用 NLTK 计算 BLEU 1-4 分数"""
    try:
        pred_tokens = nltk.word_tokenize(str(prediction).lower())
        ref_tokens = [nltk.word_tokenize(str(reference).lower())]
    except Exception:
        pred_tokens = simple_tokenize(prediction)
        ref_tokens = [simple_tokenize(reference)]

    weights_list = [
        (1, 0, 0, 0),
        (0.5, 0.5, 0, 0),
        (0.33, 0.33, 0.33, 0),
        (0.25, 0.25, 0.25, 0.25),
    ]
    smooth = SmoothingFunction().method1

    scores = {}
    for n, weights in enumerate(weights_list, start=1):
        try:
            score = sentence_bleu(
                ref_tokens,
                pred_tokens,
                weights=weights,
                smoothing_function=smooth,
            )
        except Exception:
            score = 0.0
        scores[f"bleu{n}"] = score

    return scores


def parse_locomo_results(data):
    """解析 LoCoMo 数据集结构[cite: 11]"""
    items = []
    source_list = data.get("results") or data.get("detailed_results") or []

    for item in source_list:
        cat = item.get("category") or item.get("question_type") or "uncategorized"
        cat_key = f"Category {cat}"
        pred = item.get("prediction", "")
        ref = item.get("reference", "")

        judge_score = None
        if "metrics" in item and "judge_correct" in item["metrics"]:
            judge_score = float(item["metrics"]["judge_correct"])
        elif "correct" in item:
            judge_score = float(item["correct"])

        items.append(
            {
                "category": cat_key,
                "prediction": pred,
                "reference": ref,
                "judge_correct": judge_score,
                "token_usage": item.get("token_usage", {}),
            }
        )
    return items


def parse_longmemeval_results(data):
    """解析 LongMemEval 数据集结构[cite: 10]"""
    items = []
    if "generated_answer" in data and "ground_truth" in data:
        cat_key = "LongMemEval QA"
        pred = data.get("generated_answer", "")
        ref = data.get("ground_truth", "")

        judge_score = None
        if "correct" in data and data["correct"] is not None:
            judge_score = float(data["correct"])

        items.append(
            {
                "category": cat_key,
                "prediction": pred,
                "reference": ref,
                "judge_correct": judge_score,
                "token_usage": {
                    "completion_tokens": data.get("completion_tokens", 0),
                    "prompt_tokens": data.get("prompt_tokens", 0),
                },
            }
        )
    return items


def parse_halumem_results(data):
    """解析 HaLuMem 数据集结构"""
    items = []
    # data 可能是一个列表
    if isinstance(data, list):
        for item in data:
            cat = item.get("question_type", "uncategorized")
            cat_key = f"HaMem {cat}"
            pred = item.get("answer", "")
            ref = item.get("reference", "")

            judge_score = None
            if "metrics" in item and "llm_judge_score" in item["metrics"]:
                judge_score = float(item["metrics"]["llm_judge_score"])

            # 构建 token_usage 字典，包含 answer 和 build_memory tokens
            token_usage = {
                "prompt_tokens": item.get("answer_prompt_tokens", 0),
                "completion_tokens": item.get("answer_completion_tokens", 0),
                "build_memory_prompt_tokens": item.get("turn_build_memory_prompt_tokens", 0),
                "build_memory_completion_tokens": item.get("turn_build_memory_completion_tokens", 0),
            }

            items.append({
                "category": cat_key,
                "prediction": pred,
                "reference": ref,
                "judge_correct": judge_score,
                "token_usage": token_usage,
            })
    return items


def process_eval_dataset(token_dir: str, result_dir: str, dataset_name: str):
    token_folder = Path(token_dir)
    result_folder = Path(result_dir)

    # 1. 统计 Token 消耗
    all_prompt_tokens = []
    all_completion_tokens = []

    if token_folder.exists():
        for file in token_folder.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                prompt_tokens = 0
                completion_tokens = 0

                if "llm" in data and isinstance(data["llm"], dict):
                    for item in data["llm"].values():
                        prompt_tokens += item.get("prompt_tokens", 0)
                        completion_tokens += item.get("completion_tokens", 0)
                else:
                    prompt_tokens = data.get("prompt_tokens", 0)
                    completion_tokens = data.get("completion_tokens", 0)

                if prompt_tokens > 0:
                    all_prompt_tokens.append(prompt_tokens)
                if completion_tokens > 0:
                    all_completion_tokens.append(completion_tokens)
            except Exception as e:
                print(f"Error reading token file {file}: {e}")
    else:
        print(f"Warning: Token directory {token_dir} does not exist.")

    # 2. 统计评测指标 (F1, Accuracy, BLEU 1-4)
    metrics_by_category = defaultdict(
        lambda: {
            "f1": [],
            "judge_correct": [],
            "bleu1": [],
            "bleu2": [],
            "bleu3": [],
            "bleu4": [],
        }
    )
    global_f1s = []
    global_judge_scores = []
    global_bleus = defaultdict(list)

    question_prompt_tokens = []
    question_completion_tokens = []
    build_memory_prompt_tokens = []
    build_memory_completion_tokens = []
    if result_folder.exists():
        for file_path in result_folder.glob("**/*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if "generated_answer" in data:
                    parsed_items = parse_longmemeval_results(data)
                elif isinstance(data, list) and len(data) > 0 and "turn_build_memory_prompt_tokens" in data[0]:
                    parsed_items = parse_halumem_results(data)
                else:
                    parsed_items = parse_locomo_results(data)

                for item in parsed_items:
                    cat_key = item["category"]
                    pred = item["prediction"]
                    ref = item["reference"]
                    judge_score = item["judge_correct"]
                    token_usage = item["token_usage"]
                    if token_usage:
                        # 收集问题回答的token消耗
                        if "prompt_tokens" in token_usage:
                            question_prompt_tokens.append(token_usage["prompt_tokens"])
                        if "completion_tokens" in token_usage:
                            question_completion_tokens.append(token_usage["completion_tokens"])
                        # 收集构建记忆的token消耗
                        if "build_memory_prompt_tokens" in token_usage:
                            build_memory_prompt_tokens.append(token_usage["build_memory_prompt_tokens"])
                        if "build_memory_completion_tokens" in token_usage:
                            build_memory_completion_tokens.append(token_usage["build_memory_completion_tokens"])

                    # 计算 F1
                    f1_score = compute_f1(pred, ref)
                    metrics_by_category[cat_key]["f1"].append(f1_score)
                    global_f1s.append(f1_score)

                    # 计算 BLEU 1-4
                    bleu_scores = calculate_bleu_scores(pred, ref)
                    for b_key in ["bleu1", "bleu2", "bleu3", "bleu4"]:
                        b_val = bleu_scores.get(b_key, 0.0)
                        metrics_by_category[cat_key][b_key].append(b_val)
                        global_bleus[b_key].append(b_val)

                    # 记录 Accuracy (Judge Score)
                    if judge_score is not None:
                        metrics_by_category[cat_key]["judge_correct"].append(
                            judge_score
                        )
                        global_judge_scores.append(judge_score)

            except Exception as e:
                print(f"Error reading result file {file_path}: {e}")
    else:
        print(f"Warning: Result directory {result_dir} does not exist.")

    # 3. 打印结果
    avg_prompt = np.mean(all_prompt_tokens) if all_prompt_tokens else 0.0
    avg_comp = (
        np.mean(all_completion_tokens) if all_completion_tokens else 0.0
    )

    print(f"\n================ [{dataset_name}] Summary ================")
    print(
        f"[Build] Average Prompt Tokens    : {avg_prompt:.2f}\n"
        f"[Build] Average Completion Tokens: {avg_comp:.2f}"
    )

    if len(question_prompt_tokens) > 0:
        question_prompt_tokens_avg = sum(question_prompt_tokens) / len(question_prompt_tokens)
        question_completion_tokens_avg = sum(question_completion_tokens) / len(question_completion_tokens)

        print(
            f"[QUESTION] Average Prompt Tokens    : {question_prompt_tokens_avg:.2f}\n"
            f"[QUESTION] Average Completion Tokens: {question_completion_tokens_avg:.2f}"
        )

    if len(build_memory_prompt_tokens) > 0:
        build_memory_prompt_avg = sum(build_memory_prompt_tokens) / len(build_memory_prompt_tokens)
        build_memory_completion_avg = sum(build_memory_completion_tokens) / len(build_memory_completion_tokens)

        print(
            f"[Build Memory] Average Prompt Tokens    : {build_memory_prompt_avg:.2f}\n"
            f"[Build Memory] Average Completion Tokens: {build_memory_completion_avg:.2f}"
        )

    if metrics_by_category:
        has_judge = len(global_judge_scores) > 0

        header = f"{'Category / Type':<22} | {'F1':<8} | {'BLEU-1':<8} | {'BLEU-2':<8} | {'BLEU-3':<8} | {'BLEU-4':<8}"
        if has_judge:
            header += f" | {'Accuracy':<10}"
        header += f" | {'Count':<6}"

        print("\n" + "-" * len(header))
        print(header)
        print("-" * len(header))

        for cat_key in sorted(metrics_by_category.keys()):
            cat_data = metrics_by_category[cat_key]

            avg_f1 = np.mean(cat_data["f1"]) if cat_data["f1"] else 0.0
            avg_b1 = np.mean(cat_data["bleu1"]) if cat_data["bleu1"] else 0.0
            avg_b2 = np.mean(cat_data["bleu2"]) if cat_data["bleu2"] else 0.0
            avg_b3 = np.mean(cat_data["bleu3"]) if cat_data["bleu3"] else 0.0
            avg_b4 = np.mean(cat_data["bleu4"]) if cat_data["bleu4"] else 0.0

            line = (
                f"{str(cat_key):<22} | {avg_f1:<8.4f} | {avg_b1:<8.4f} | "
                f"{avg_b2:<8.4f} | {avg_b3:<8.4f} | {avg_b4:<8.4f}"
            )

            if has_judge:
                judges = cat_data["judge_correct"]
                avg_judge = np.mean(judges) if judges else None
                judge_str = (
                    f"{avg_judge:<10.4f}"
                    if avg_judge is not None
                    else f"{'N/A':<10}"
                )
                line += f" | {judge_str}"

            line += f" | {len(cat_data['f1']):<6}"
            print(line)

        # 输出 OVERALL 总平均
        print("-" * len(header))
        ov_f1 = np.mean(global_f1s) if global_f1s else 0.0
        ov_b1 = np.mean(global_bleus["bleu1"]) if global_bleus["bleu1"] else 0.0
        ov_b2 = np.mean(global_bleus["bleu2"]) if global_bleus["bleu2"] else 0.0
        ov_b3 = np.mean(global_bleus["bleu3"]) if global_bleus["bleu3"] else 0.0
        ov_b4 = np.mean(global_bleus["bleu4"]) if global_bleus["bleu4"] else 0.0

        ov_line = (
            f"{'OVERALL (Total Avg)':<22} | {ov_f1:<8.4f} | {ov_b1:<8.4f} | "
            f"{ov_b2:<8.4f} | {ov_b3:<8.4f} | {ov_b4:<8.4f}"
        )

        if has_judge:
            overall_judge = (
                np.mean(global_judge_scores) if global_judge_scores else 0.0
            )
            ov_line += f" | {overall_judge:<10.4f}"

        ov_line += f" | {len(global_f1s):<6}"
        print(ov_line)
        print("=" * len(header))
    else:
        print("No metrics found.")


if __name__ == "__main__":
    tasks = [
        (
            "./experiments/token_consumption_build_memory_locomo_event",
            "./experiments/lightmem_locomo_results",
            "locomo",
        ),
        (
            "./experiments/token_consumption_build_memory_longmemeval_event",
            "./experiments/lightmem_longmemeval_results_event",
            "longmemeval",
        ),
        (
            "./experiments/token_consumption_build_memory_halumem_event",
            "./experiments/lightmem_halumem_results_event",
            "halumem",
        ),
    ]

    for token_path, result_path, name in tasks:
        process_eval_dataset(token_path, result_path, name)