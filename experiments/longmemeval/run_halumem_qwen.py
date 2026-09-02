import os
import time
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from multiprocessing import Pool
import sys

# 从 lightmem 框架导入 LightMemory
from lightmem.memory.lightmem import LightMemory

# ============================================================================
# API 配置与模型参数
# ============================================================================

JUDGE_MODEL_API_KEY = os.environ.get('JUDGE_API_KEY', 'sk-11ce7640e46049a6977c0d96ba855ffb')
JUDGE_MODEL_BASE_URL = os.environ.get('JUDGE_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
JUDGE_MODEL = 'deepseek-v3.2'

API_KEY = os.environ.get('LLM_API_KEY', 'sk-dummy')
API_BASE_URL = os.environ.get('LLM_BASE_URL', 'http://127.0.0.1:30004/v1')
LLM_MODEL = os.environ.get('LLM_MODEL', 'qwen3-8b')

LLMLINGUA_MODEL_PATH = os.environ.get('LLMLINGUA_MODEL_PATH', '/mnt/qjhs-sh-lab-01/models/llmlingua-2-bert-base-multilingual-cased-meetingbank')
EMBEDDING_MODEL_PATH = os.environ.get('EMBEDDING_MODEL_PATH', '/mnt/qjhs-sh-lab-01/models/all-MiniLM-L6-v2')


# ============================================================================
# HaluMem 数据结构与 JSONL 加载器
# ============================================================================

@dataclass
class HaluMemSample:
    sample_id: str
    persona_info: str
    sessions: List[Dict]


from datetime import datetime


def format_timestamp_for_lightmem(raw_time: str) -> str:
    """
    将 HaluMem 的时间格式转换为 LightMem 兼容的格式
    输入示例: 'Sep 04, 2025, 12:07:43'
    输出示例: '2025/09/04 (Thu) 12:07'
    """
    if not raw_time:
        # 如果原始时间为空，提供默认时间占位
        return "2025/01/01 (Wed) 00:00"

    # 尝试解析 HaluMem 常见的几类时间格式
    patterns = [
        "%b %d, %Y, %H:%M:%S",  # 'Sep 04, 2025, 12:07:43'
        "%b %d, %Y, %H:%M",  # 'Sep 04, 2025, 12:07'
        "%Y-%m-%d %H:%M:%S",  # '2025-09-04 12:07:43'
        "%Y-%m-%d %H:%M",  # '2025-09-04 12:07'
    ]

    dt = None
    for pattern in patterns:
        try:
            dt = datetime.strptime(str(raw_time).strip(), pattern)
            break
        except ValueError:
            continue

    if dt is None:
        # 如果解析失败，回退到默认时间，防止程序 crash
        return "2025/01/01 (Wed) 00:00"

    # 格式化为 LightMem 期望的 'YYYY/MM/DD (Wed) HH:MM'
    return dt.strftime("%Y/%m/%d (%a) %H:%M")


def load_halumem_dataset(path_input: Union[str, Path]) -> List[HaluMemSample]:
    """
    加载 HaluMem 数据集 (.jsonl 格式或包含 jsonl 的目录)
    """
    path_input = Path(path_input)
    if not path_input.exists():
        raise FileNotFoundError(f"Path not found at {path_input}")

    files = []
    if path_input.is_dir():
        files = sorted(list(path_input.glob("*.jsonl")))
    else:
        files = [path_input]

    samples = []
    for fpath in files:
        with open(fpath, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                sid = data.get("uuid", f"{fpath.stem}_{line_idx}")
                persona = data.get("persona_info", "")
                sessions = data.get("sessions", [])
                samples.append(HaluMemSample(sample_id=str(sid), persona_info=persona, sessions=sessions))

    print(f"Successfully loaded {len(samples)} samples from {path_input}")
    return samples


# ============================================================================
# LLM 客户端与评测辅助函数
# ============================================================================

class LLMModel:
    def __init__(self, model_name: str, api_key: str, base_url: str):
        self.name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.max_tokens = 2000
        self.temperature = 0.0
        self.top_p = 0.8
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def call_with_tokens(self, messages: list, **kwargs):
        max_retries = kwargs.get("max_retries", 3)
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    stream=False,
                    extra_body={
                        "chat_template_kwargs": {
                            "enable_thinking": False,
                            "thinking": False
                        }
                    },
                )
                response = completion.choices[0].message.content
                prompt_tokens = getattr(completion.usage, 'prompt_tokens', 0)
                completion_tokens = getattr(completion.usage, 'completion_tokens', 0)
                return response, prompt_tokens, completion_tokens
            except Exception as e:
                print(f"[Retry {attempt + 1}/{max_retries}] {type(e).__name__}: {e}")
                if attempt == max_retries - 1:
                    raise

    def call(self, messages: list, **kwargs):
        res, _, _ = self.call_with_tokens(messages, **kwargs)
        return res


def get_anscheck_prompt(question: str, reference: str, response: str) -> str:
    template = (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
        "If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, "
        "you should also answer yes. If the response only contains a subset of the information required by the answer, answer no.\n\n"
        "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
        "Is the model response correct? Answer yes or no only."
    )
    return template.format(question, reference, response)


def parse_llm_judge_response(response: str) -> bool:
    if not response:
        return False
    normalized = str(response).strip().lower()
    if not normalized:
        return False
    first_line = normalized.splitlines()[0].strip()
    tokens = first_line.replace('.', '').replace('!', '').replace(':', '').replace(';', '').split()
    if not tokens:
        return False
    head = tokens[0]
    if head in ("yes", "y"):
        return True
    if head in ("no", "n"):
        return False
    return "yes" in first_line


def calculate_simple_f1(prediction: str, reference: str) -> float:
    """简单的 Token 级 F1 计算辅助函数"""
    pred_tokens = prediction.strip().lower().split()
    ref_tokens = reference.strip().lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * (precision * recall) / (precision + recall)


# ============================================================================
# LightMemory 工厂函数
# ============================================================================

def build_lightmem_instance(collection_name: str, qdrant_dir: str, extraction_mode: str = "flat") -> LightMemory:
    config = {
        "pre_compress": True,
        "pre_compressor": {
            "model_name": "llmlingua-2",
            "configs": {
                "llmlingua_config": {
                    "model_name": LLMLINGUA_MODEL_PATH,
                    "device_map": "cuda",
                    "use_llmlingua2": True,
                },
            }
        },
        "topic_segment": True,
        "precomp_topic_shared": True,
        "topic_segmenter": {
            "model_name": "llmlingua-2",
        },
        "messages_use": "user_only",
        "metadata_generate": True,
        "text_summary": True,
        "memory_manager": {
            "model_name": "openai",
            "configs": {
                "model": LLM_MODEL,
                "api_key": API_KEY,
                "max_tokens": 16000,
                "openai_base_url": API_BASE_URL
            }
        },
        "extract_threshold": 0.1,
        "index_strategy": "embedding",
        "text_embedder": {
            "model_name": "huggingface",
            "configs": {
                "model": EMBEDDING_MODEL_PATH,
                "embedding_dims": 384,
                "model_kwargs": {"device": "cuda"},
            },
        },
        "retrieve_strategy": "embedding",
        "embedding_retriever": {
            "model_name": "qdrant",
            "configs": {
                "collection_name": collection_name,
                "embedding_model_dims": 384,
                "path": os.path.join(qdrant_dir, collection_name),
                "on_disk": True,
            }
        },
        "update": "offline",
    }
    if extraction_mode == "event":
        config = {
            **config,
            "summary_retriever": {
                "model_name": "qdrant",
                "configs": {
                    "collection_name": f"{collection_name}_summary",
                    "embedding_model_dims": 384,
                    "path": os.path.join(qdrant_dir, f"{collection_name}_summary"),
                    "on_disk": True,
                }
            },
            "extraction_mode": extraction_mode
        }
    return LightMemory.from_config(config)


# ============================================================================
# HaluMem + LightMem 增量测试器
# ============================================================================

class HaluMemLightMemTester:
    def __init__(self, use_llm_judge: bool = False, extraction_mode: str = "flat",
                 qdrant_dir: str = "./qdrant_data"):
        self.use_llm_judge = use_llm_judge
        self.extraction_mode = extraction_mode
        self.qdrant_dir = qdrant_dir
        self.llm = LLMModel(LLM_MODEL, API_KEY, API_BASE_URL)
        if use_llm_judge:
            self.judge_client = LLMModel(JUDGE_MODEL, JUDGE_MODEL_API_KEY, JUDGE_MODEL_BASE_URL)
        else:
            self.judge_client = None

    def run_single_sample(self, sample: HaluMemSample, sample_idx: int, save_dir: str = "./results_lightmem_halumem"):
        safe_sample_id = sample.sample_id.replace(" ", "_")
        collection_name = f"halumem_{safe_sample_id[:20]}"

        # 初始化 LightMemory
        lightmem = build_lightmem_instance(collection_name, self.qdrant_dir, self.extraction_mode)

        sample_results = []
        accumulated_history_turns = 0
        total_build_prompt_tokens = 0
        total_build_completion_tokens = 0

        # 遍历各个 session
        for s_idx, session in enumerate(sample.sessions):
            print(f"running {sample_idx} {s_idx}/{len(sample.sessions)}")
            dialogue_turns = session.get("dialogue", [])
            questions = session.get("questions", [])

            # ----------------------------------------------------------------
            # 步骤 1：将当前 Session 的对话增量写入 LightMem
            # ----------------------------------------------------------------
            stats_before = lightmem.get_token_statistics() if hasattr(lightmem, 'get_token_statistics') else {}

            num_turns = len(dialogue_turns) // 2
            turns_added_count = 0

            for turn_idx in range(num_turns):
                turn_messages = dialogue_turns[turn_idx * 2: turn_idx * 2 + 2]
                if len(turn_messages) < 2:
                    continue
                turn_messages[0]["time_stamp"] = format_timestamp_for_lightmem(turn_messages[0]["timestamp"])
                turn_messages[1]["time_stamp"] = format_timestamp_for_lightmem(turn_messages[1]["timestamp"])

                is_last_turn = turn_idx == num_turns - 1
                if is_last_turn:
                    print("last turn")
                lightmem.add_memory(
                    messages=turn_messages,
                    force_segment=is_last_turn,
                    force_extract=is_last_turn
                )
                turns_added_count += 1

            accumulated_history_turns += turns_added_count

            # 统计本轮 Build Memory 产生的 Token 增量
            stats_after = lightmem.get_token_statistics() if hasattr(lightmem, 'get_token_statistics') else {}

            turn_build_prompt_tokens_after = stats_after["llm"]["add_memory"]["prompt_tokens"] + stats_after["llm"]["update"]["prompt_tokens"] + stats_after["llm"]["summarize"]["prompt_tokens"]
            turn_build_completion_tokens_after = stats_after["llm"]["add_memory"]["completion_tokens"] + stats_after["llm"]["update"]["completion_tokens"] + stats_after["llm"]["summarize"]["completion_tokens"]
            turn_build_prompt_tokens_before = stats_before["llm"]["add_memory"]["prompt_tokens"] + stats_before["llm"]["update"]["prompt_tokens"] + stats_before["llm"]["summarize"]["prompt_tokens"]
            turn_build_completion_tokens_before = stats_before["llm"]["add_memory"]["completion_tokens"] + stats_before["llm"]["update"]["completion_tokens"] + stats_before["llm"]["summarize"]["completion_tokens"]

            turn_build_prompt_tokens = turn_build_prompt_tokens_after - turn_build_prompt_tokens_before
            turn_build_completion_tokens = turn_build_completion_tokens_after - turn_build_completion_tokens_before

            total_build_prompt_tokens = turn_build_prompt_tokens_after
            total_build_completion_tokens = turn_build_completion_tokens_after

            # ----------------------------------------------------------------
            # 步骤 2：对当前 Session 内的问题进行检索与回答
            # ----------------------------------------------------------------
            for q_idx, q_item in enumerate(questions):
                question = q_item.get("question", "")
                reference = str(q_item.get("answer", ""))
                question_type = q_item.get("question_type", "default")
                question_id = f"{safe_sample_id}_s{s_idx}_q{q_idx}"

                # 2.1 检索阶段
                retrieval_start = time.time()
                related_memories = lightmem.retrieve(question, limit=20)
                retrieval_time = time.time() - retrieval_start

                # LightMem 检索阶段在本地向量库计算，无 LLM API Token 消耗
                retrieval_prompt_tokens = 0
                retrieval_completion_tokens = 0

                # 2.2 回答生成阶段
                answer_start = time.time()
                memory_text = "\n".join(related_memories) if isinstance(related_memories, list) else str(related_memories)
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Question: {question}\nPlease answer the question based on the following memories:\n{memory_text}"
                    }
                ]
                generated_answer, answer_prompt_tokens, answer_completion_tokens = self.llm.call_with_tokens(messages)
                answer_time = time.time() - answer_start

                # 2.3 指标计算
                f1_score = calculate_simple_f1(generated_answer, reference)
                llm_judge_score = 0.0

                if self.use_llm_judge and self.judge_client:
                    judge_prompt = get_anscheck_prompt(question, reference, generated_answer)
                    judge_response = self.judge_client.call([{"role": "user", "content": judge_prompt}])
                    llm_judge_score = 1.0 if parse_llm_judge_response(judge_response) else 0.0

                metrics = {
                    "f1": f1_score,
                    "llm_judge_score": llm_judge_score
                }

                query_res = {
                    'sample_id': safe_sample_id,
                    'question_id': question_id,
                    'session_index': s_idx,
                    'question_type': question_type,
                    'difficulty': q_item.get("difficulty", "normal"),
                    'question': question,
                    'answer': generated_answer,
                    'reference': reference,
                    'related_memories': related_memories,

                    # 1. 本轮及累计记忆构建 Token 开销
                    'session_dialogue_turns_added': turns_added_count,
                    'turn_build_memory_prompt_tokens': turn_build_prompt_tokens,
                    'turn_build_memory_completion_tokens': turn_build_completion_tokens,
                    'turn_build_memory_total_tokens': turn_build_prompt_tokens + turn_build_completion_tokens,

                    'accumulated_history_turns': accumulated_history_turns,
                    'total_build_memory_prompt_tokens': total_build_prompt_tokens,
                    'total_build_memory_completion_tokens': total_build_completion_tokens,

                    # 2. 检索阶段 Token 开销
                    'retrieval_prompt_tokens': retrieval_prompt_tokens,
                    'retrieval_completion_tokens': retrieval_completion_tokens,
                    'retrieval_total_tokens': 0,

                    # 3. 回答生成阶段 Token 开销
                    'answer_prompt_tokens': answer_prompt_tokens,
                    'answer_completion_tokens': answer_completion_tokens,
                    'answer_total_tokens': answer_prompt_tokens + answer_completion_tokens,

                    # 4. QA 环节总 Token 消耗
                    'query_total_prompt_tokens': retrieval_prompt_tokens + answer_prompt_tokens,
                    'query_total_completion_tokens': retrieval_completion_tokens + answer_completion_tokens,
                    'query_total_tokens': (retrieval_prompt_tokens + answer_prompt_tokens) + (retrieval_completion_tokens + answer_completion_tokens),

                    'retrieval_time': retrieval_time,
                    'answer_time': answer_time,
                    'total_time': retrieval_time + answer_time,
                    'num_retrieved': len(related_memories) if isinstance(related_memories, list) else 1,
                    'metrics': metrics
                }
                sample_results.append(query_res)

            # 保存单个 sample 结果
            os.makedirs(save_dir, exist_ok=True)
            with open(f"{save_dir}/{safe_sample_id}.json", 'w', encoding='utf-8') as f:
                json.dump(sample_results, f, indent=2, ensure_ascii=False)

        avg_f1 = sum(r['metrics']['f1'] for r in sample_results) / len(sample_results) if sample_results else 0
        print(f"[{sample_idx}] Sample ID: {safe_sample_id} | Queries: {len(sample_results)} | Avg F1: {avg_f1:.3f}")
        return sample_results


# ============================================================================
# 多进程 Worker 定义 (必须放在全局位置，以便 pickle 序列化)
# ============================================================================

def _worker(args_tuple):
    idx, sample, args = args_tuple
    tester = HaluMemLightMemTester(
        use_llm_judge=args.llm_judge,
        extraction_mode=args.extraction_mode,
        qdrant_dir=args.qdrant_dir
    )
    return tester.run_single_sample(sample, idx, save_dir=args.output_dir)


# ============================================================================
# 主函数入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run LightMem on HaluMem dataset')
    parser.add_argument('--dataset', type=str, default='../../data/HaluMem-Medium.jsonl',
                        help='Path to HaluMem jsonl file or directory')
    parser.add_argument('--extraction_mode', type=str, default='flat', choices=['flat', 'event'],
                        help='Extraction mode for LightMem')
    parser.add_argument('--llm-judge', action='store_true', help='Enable LLM-as-judge evaluation')
    parser.add_argument('--output-dir', type=str, default='../lightmem_halumem_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--qdrant-dir', type=str, default='./qdrant_data_halumem',
                        help='Directory to store Qdrant database files')
    args = parser.parse_args()

    samples = load_halumem_dataset(args.dataset)

    if args.extraction_mode == "event":
        args.output_dir = f"{args.output_dir}_event"
        args.qdrant_dir = f"{args.qdrant_dir}_event"

    max_workers = 16
    if os.environ.get('DEBUG') == "1":
        max_workers = 1

    # 打包任务参数传给全局 _worker
    tasks = [(i, s, args) for i, s in enumerate(samples)]

    all_flattened_results = []
    try:
        with Pool(processes=max_workers) as pool:
            results = pool.map(_worker, tasks)
            for sample_res in results:
                if sample_res:
                    all_flattened_results.extend(sample_res)
    except KeyboardInterrupt:
        print("\n[Ctrl+C] 收到中断信号，正在终止所有子进程...")
        pool.terminate()
        pool.join()
        sys.exit(1)

    if all_flattened_results:
        avg_f1 = sum(r['metrics']['f1'] for r in all_flattened_results) / len(all_flattened_results)
        print("\n" + "=" * 80)
        print(" HaluMem + LightMem Test Summary ".center(80, "="))
        print(f"Total Queries Evaluated Across All Samples: {len(all_flattened_results)}")
        print(f"Overall Average F1: {avg_f1:.4f}")
        if args.llm_judge:
            avg_judge = sum(r['metrics']['llm_judge_score'] for r in all_flattened_results) / len(all_flattened_results)
            print(f"Overall LLM Judge Acc: {avg_judge:.4f}")
        print("=" * 80)


if __name__ == "__main__":
    main()