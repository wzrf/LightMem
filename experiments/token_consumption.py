import json
from collections import defaultdict
from pathlib import Path
import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import numpy as np
import os
import re
import hashlib
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI


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


# ========== LLM Judge Functions ==========

SYSTEM_PROMPT = """You are a strict, method-blind evaluator of question answering. Judge only whether the candidate answer is semantically correct according to the question and reference answer. Do not infer which system produced it."""

JUDGE_TEMPLATE = """Decide whether the candidate answer is correct.

Rules:
1. Accept concise paraphrases, equivalent names, equivalent date/number formats, and a correct answer embedded in harmless extra explanation.
2. Reject a wrong person, entity, event, date, ordering, count, amount, or polarity; a contradiction; a refusal when the reference answers the question; or an answer missing a required list item, comparison, calculation, or event.
3. Extra text is harmless only if it does not add a materially false answer claim.
4. For open-ended preference or recommendation questions, the answer need not copy every example in the reference, but it must correctly use the core personal information required by the reference.
5. Treat the reference as the scoring ground truth. Do not use outside knowledge.

Question:
{question}

Reference answer:
{reference}

Candidate answer:
{prediction}

Do not REASON. JUST GIVE THE RESULT.
Return exactly one JSON object with one boolean field and no other text:
{{"correct": true}}
or
{{"correct": false}}"""


# ========== Cache Management ==========

def get_cache_path():
    """获取缓存目录路径"""
    cache_dir = Path("./.llm_judge_cache")
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


def compute_cache_key(question: str, prediction: str, reference: str,
                      model: str = "gpt-4o-mini",
                      prompt_version: str = "v1") -> str:
    """
    计算缓存键，基于问题、答案、参考答案、模型和prompt版本
    返回：SHA256哈希字符串
    """
    # 使用prompt模板和内容计算哈希
    content_to_hash = f"{SYSTEM_PROMPT}||{JUDGE_TEMPLATE}||{prompt_version}||{model}||{question}||{prediction}||{reference}"
    return hashlib.sha256(content_to_hash.encode('utf-8')).hexdigest()


def load_from_cache(cache_key: str):
    """从缓存加载结果"""
    cache_dir = get_cache_path()
    cache_file = cache_dir / f"{cache_key}.json"

    if cache_file.exists():
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 验证缓存数据格式
                if "score" in data and "timestamp" in data:
                    return data["score"]
        except Exception as e:
            print(f"Warning: Failed to load cache {cache_key}: {e}")
    return None


def save_to_cache(cache_key: str, score: float):
    """保存结果到缓存"""
    try:
        cache_dir = get_cache_path()
        cache_file = cache_dir / f"{cache_key}.json"

        cache_data = {
            "score": score,
            "timestamp": int(time.time()),
            "key": cache_key
        }

        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save cache {cache_key}: {e}")


def llm_judge_answer(question: str, prediction: str, reference: str,
                      model: str = None,
                      api_key: str = None,
                      base_url: str = None,
                      timeout: int = 30,
                      prompt_version: str = "v1") -> float:
    """
    使用LLM判断预测答案是否正确
    返回：1.0（正确）或0.0（错误）
    """
    # 获取配置，优先使用传入参数，其次使用环境变量，最后使用默认值
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "sk-dummy")
    base_url = base_url or os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:30002/v1")
    model = model or os.environ.get("OPENAI_MODEL", "GLM-5.3")

    # 计算缓存键
    cache_key = compute_cache_key(question, prediction, reference, model, prompt_version)

    # 检查缓存
    cached_score = load_from_cache(cache_key)
    if cached_score is not None:
        return cached_score

    try:
        # 获取API密钥，优先使用传入的参数，其次使用环境变量
        if not api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")

        # 创建同步客户端
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

        # 格式化prompt
        prompt = JUDGE_TEMPLATE.format(
            question=question,
            reference=reference,
            prediction=prediction
        )

        # 调用API
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2048,
            stream=False,
            response_format={"type": "json_object"},
            model=model,
            extra_body={"chat_template_kwargs": {"reasoning_effort": "low"}}
        )

        # 解析响应
        content = response.choices[0].message.content or ""

        # 尝试解析JSON
        text = str(content or "").strip()
        text = re.sub(r"^(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*$", "", text)

        try:
            value = json.loads(text)
        except Exception:
            raise ValueError(f"Failed to parse JSON response: {text[:100]}")

        if not isinstance(value, dict) or set(value) != {"correct"} or not isinstance(value["correct"], bool):
            raise ValueError("judge response is not strict correct:boolean JSON")

        # 计算最终分数
        score = 1.0 if value["correct"] else 0.0

        # 保存到缓存
        save_to_cache(cache_key, score)

        # 返回正确分数：正确为1.0，错误为0.0
        return score

    except Exception as e:
        # 记录错误并返回默认值0.0
        print(f"LLM judge error for question '{question}...': {e}")
        return 0.0


def batch_llm_judge(items_to_judge, model="GLM-5.3", max_concurrent=32, prompt_version="v1"):
    """
    批量处理需要LLM判断的项目 - 多线程版本
    """
    final_scores = []

    def process_item(item):
        try:
            question = item.get("question", "")
            return llm_judge_answer(
                question=question,
                prediction=item["prediction"],
                reference=item["reference"],
                model=model,
                prompt_version=prompt_version
            )
        except Exception as e:
            print(f"Error judging item: {e}")
            return 0.0  # 错误时默认0分

    # 使用线程池并发处理
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        # 提交所有任务
        future_to_item = {
            executor.submit(process_item, item): idx
            for idx, item in enumerate(items_to_judge)
        }

        # 按完成顺序收集结果
        for future in as_completed(future_to_item):
            idx = future_to_item[future]
            try:
                result = future.result()
                final_scores.append((idx, result))
            except Exception as e:
                print(f"Error processing item {idx}: {e}")
                final_scores.append((idx, 0.0))

    # 按原始顺序排序并返回结果
    final_scores.sort(key=lambda x: x[0])
    return [score for idx, score in final_scores]




def parse_locomo_results(data):
    """解析 LoCoMo 数据集结构[cite: 11]"""
    items = []
    source_list = data.get("results") or data.get("detailed_results") or []

    for item in source_list:
        cat = item.get("category") or item.get("question_type") or "uncategorized"
        cat_key = f"Category {cat}"
        pred = item.get("prediction", "")
        ref = item.get("reference", "")
        question = item.get("question", "")  # 提取question字段

        if pred.startswith("Answer: "):
            pred = pred[len("Answer: ") :]

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
                "question": question,  # 添加question字段
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
                "question": data.get("question", ""),  # LongMemEval可能没有question字段
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
                "question": item.get("question", ""),  # 添加question字段
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
                        if isinstance(item, dict):
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

    # 收集需要LLM判断的项目
    items_need_judge = []  # 存储需要判断的项目信息
    judge_item_positions = []  # 存储项目位置：(category_key, 其他信息用于更新统计)

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
                    # 忽略原有的judge_score，总是添加None占位符，然后进行LLM判断
                    # 将None添加到judge_correct列表作为占位符
                    metrics_by_category[cat_key]["judge_correct"].append(None)

                    # 总是收集项目进行LLM判断（缓存会避免重复计算）
                    items_need_judge.append({
                        "question": item.get("question", ""),
                        "prediction": pred,
                        "reference": ref,
                    })
                    # 记录位置信息：category和judge_correct列表中的索引
                    judge_item_positions.append({
                        "category": cat_key,
                        "judge_index": len(metrics_by_category[cat_key]["judge_correct"]) - 1,  # 刚添加的占位符索引
                    })

            except Exception as e:
                print(f"Error reading result file {file_path}: {e}")
    else:
        print(f"Warning: Result directory {result_dir} does not exist.")

    # 3. LLM判断处理 - 对所有项目进行判断（使用缓存避免重复）
    if items_need_judge:
        print(f"\nRunning LLM judge for {len(items_need_judge)} items...")
        try:
            judged_scores = batch_llm_judge(items_need_judge, model="GLM-5.3", max_concurrent=32)

            # 更新统计中的占位符
            for idx, score in enumerate(judged_scores):
                if idx < len(judge_item_positions):
                    pos = judge_item_positions[idx]
                    cat_key = pos["category"]
                    judge_index = pos["judge_index"]

                    # 更新占位符
                    metrics_by_category[cat_key]["judge_correct"][judge_index] = score
                    # 添加到全局统计
                    global_judge_scores.append(score)

            print(f"LLM judge completed. Updated {len(judged_scores)} items.")
        except Exception as e:
            print(f"Error during LLM judging: {e}")
            # LLM判断失败，用0.0填充所有占位符
            for idx in range(len(items_need_judge)):
                if idx < len(judge_item_positions):
                    pos = judge_item_positions[idx]
                    cat_key = pos["category"]
                    judge_index = pos["judge_index"]
                    metrics_by_category[cat_key]["judge_correct"][judge_index] = 0.0
                    global_judge_scores.append(0.0)
    else:
        print(f"\nNo items need LLM judging.")

    # 4. 打印结果
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
                # 过滤掉None值（LLM判断失败的情况）
                valid_judges = [j for j in judges if j is not None]
                avg_judge = np.mean(valid_judges) if valid_judges else None
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
        # (
        #     "./token_consumption_build_memory_locomo_event_fusionrag/",
        #     "./lightmem_locomo_results_event_fusionrag",
        #     "locomo",
        # ),
        # (
        #     "./token_consumption_build_memory_locomo_fusionrag/",
        #     "./lightmem_locomo_results_fusionrag",
        #     "locomo",
        # ),

        (
            "./token_consumption_build_memory_locomo/",
            "./lightmem_locomo_results",
            "locomo_lightmem_qwen3",
        ),
        (
            "./token_consumption_build_memory_locomo_event/",
            "./lightmem_locomo_results_event",
            "locomo_structmem_qwen3",
        ),
        (
            "./token_consumption_build_memory_locomo_GLM-4.5-Air",
            "./lightmem_locomo_results_GLM-4.5-Air",
            "locomo_lightmem_GLM-4.5-Air",
        ),
        (
            "./token_consumption_build_memory_locomo_event_GLM-4.5-Air",
            "./lightmem_locomo_results_event_GLM-4.5-Air",
            "locomo_Structmem_GLM-4.5-Air",
        ),
        (
            "./token_consumption_build_memory_locomo_Kimi-K2.6",
            "./lightmem_locomo_results_Kimi-K2.6",
            "locomo_lightmem_Kimi-K2.6",
        ),
        (
            "./token_consumption_build_memory_locomo_event_Kimi-K2.6",
            "./lightmem_locomo_results_event_Kimi-K2.6",
            "locomo_Structmem_Kimi-K2.6",
        ),
        (
            "./token_consumption_build_memory_longmemeval",
            "./lightmem_longmemeval_results",
            "lme_lightmem_qwen3",
        ),
        (
            "./token_consumption_build_memory_longmemeval_event",
            "./lightmem_longmemeval_results_event",
            "lme_structmem_qwen3",
        ),
        (
            "./token_consumption_build_memory_longmemeval_GLM-4.5-Air",
            "./lightmem_longmemeval_results_GLM-4.5-Air",
            "lme_lightmem_GLM-4.5-Air",
        ),
        (
            "./token_consumption_build_memory_longmemeval_event_GLM-4.5-Air",
            "./lightmem_longmemeval_results_event_GLM-4.5-Air",
            "lme_structsmem_GLM-4.5-Air",
        ),
        (
            "./token_consumption_build_memory_longmemeval_Kimi-K2.6",
            "./lightmem_longmemeval_results_Kimi-K2.6",
            "lme_lightmem_kimi-K2.6",
        ),
        (
            "./token_consumption_build_memory_longmemeval_event_Kimi-K2.6",
            "./lightmem_longmemeval_results_event_Kimi-K2.6",
            "lme_structmem_kimi-K2.6",
        ),

        # (
        #     "./token_consumption_build_memory_halumem_event",
        #     "./lightmem_halumem_results",
        #     "halumem",
        # ),
    ]

    for token_path, result_path, name in tasks:
        process_eval_dataset(token_path, result_path, name)