import copy
import json
import os
import sys
import csv
import shutil
import time
import torch
import numpy as np
from datetime import datetime
from openai import OpenAI
from transformers import AutoTokenizer, AutoConfig
from .utils import (
    find_all_substr_needs_recompute
)
import hashlib
from FlagEmbedding import FlagModel
import faiss
import requests
from typing import List, Dict, Any, Optional
from .sglang_kvcache import run_one_question_sglang


DEFAULT_SYSTEM_PROMPT = "<|im_start|>system\nYou are a helpful assistant.\nWrite a brief and high-quality answer for the given question using only the provided search results.\n"

question_test = {
    "question": """
    Who is the spouse of the Green performer? please output using json format
    please output using json format:
    {{
    "reason": "",
    "sub_query": ""
    }}
    """,
    "gold_docs": [
        "Miquette Giraudy (born 9 February 1953, Nice, France) is a keyboard player and vocalist, best known for her work in Gong and with her partner Steve Hillage. She and Hillage currently form the core of the ambient band System 7. In addition to her performances in music, she has also worked as an actress, film editor and writer. In each role, she has used different stage names.",
        "Green (Steve Hillage album): Green is the fourth studio album by British progressive rock musician Steve Hillage. Written in spring 1977 at the same time as his previous album, the funk-inflected \"Motivation Radio\" (1977), \"Green\" was originally going to be released as \"The Green Album\" as a companion to \"The Red Album\" (the originally intended name for \"Motivation Radio\"). However, this plan was dropped and after a US tour in late 1977, \"Green\" was recorded alone, primarily in Dorking, Surrey, and in London."
    ],
    "answer": "Miquette Giraudy"
}

def get_tensor_hashkey(tensor: torch.Tensor) -> str:
    return hashlib.md5(tensor.cpu().numpy().tobytes()).hexdigest()

class RerankModel:
    def __init__(self,
                 bge_model_path: str):
        self.bgem3 = FlagModel(bge_model_path, use_fp16=True, device="cuda:0")

    def preprocess_build_faiss_index(self, all_documents: list[str], topk: int):
        corpus_embeddings = self.bgem3.encode(all_documents)
        # Build FAISS index
        dim = corpus_embeddings.shape[-1]
        index = faiss.index_factory(dim, 'Flat', faiss.METRIC_INNER_PRODUCT)
        corpus_embeddings = corpus_embeddings.astype(np.float32)
        index.train(corpus_embeddings)
        index.add(corpus_embeddings)
        print(f"FAISS index built with {index.ntotal} vectors")

        # Search for similar documents globally
        print(f"Searching for top-{topk} similar documents for each document...")
        corpus_embeddings_query = self.bgem3.encode_queries(all_documents)
        corpus_embeddings_query = corpus_embeddings_query.astype(np.float32)
        score, idx = index.search(corpus_embeddings_query, k=topk)
        context_rank = idx  # Shape: [total_docs, topk]
        return context_rank

    def clean_(self):
        self.bgem3.model = self.bgem3.model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        del self.bgem3
        import gc
        gc.collect()

class FusionRAGModel:
    def __init__(
            self,
            model_path: str,
            draft_model_path: str,
            use_multi_gpu=True,
            model_type='qwen',
            draft_model_type='qwen',
            device="cuda:0",
            draft_model_device="cuda:0",
            max_cache_len=32768,
            cache_path='',
            model_name='Qwen2.5-7B-Instruct',
            draft_model_name='',
            preprocess=False,
            preprocess_method="default",
            file_input="",
            preprocess_model_path="/data2/qy_tmp/xumengyao/bge-m3",
            max_memory=None,
            use_origin_draft_model=False,
            use_local_draft_model=True,
            draft_model_url="",
            apikey="",
    ):
        if draft_model_name == '':
            print(f"draft_model_name must be specified")
            exit(1)

        print(f"init FusionRAGModel")
        self.model_name=model_name
        self.model_type = model_type
        self.draft_model_type=draft_model_type
        self.model_cache_root = os.path.join(cache_path, model_name)
        self.draft_model_cache_root = os.path.join(cache_path, draft_model_name)
        self.save_path = os.path.join(self.model_cache_root, 'kv_cache')
        self.draft_model_save_path = os.path.join(self.draft_model_cache_root, 'kv_cache')
        self.preprocess_save_path = os.path.join(self.model_cache_root, 'preprocess_kv_cache')
        self.draft_model_preprocess_save_path = os.path.join(self.draft_model_cache_root, 'preprocess_kv_cache')
        self.preprocess_empty_prefix_save_path = os.path.join(self.model_cache_root, 'empty_prefix_preprocess_kv_cache')
        self.preprocess_method=preprocess_method
        self.api_key = apikey
        self.file_input = file_input
        if cache_path != "":
            os.makedirs(self.save_path, exist_ok=True)
            os.makedirs(self.preprocess_save_path, exist_ok=True)
            os.makedirs(self.preprocess_empty_prefix_save_path, exist_ok=True)
        self.preprocess=preprocess
        print(f"file_input={file_input}")
        dataset_name = os.path.basename(file_input).split(".")[0]
        self.dataset_name = dataset_name
        if preprocess and self.preprocess_method == "default":
            similar_index_save_path = os.path.join(self.model_cache_root, "similar_index")
            self.similar_index_file_path = os.path.join(similar_index_save_path, f"{dataset_name}.npy")
            print(f"self.similar_index_file_path = {self.similar_index_file_path}")
            os.makedirs(similar_index_save_path, exist_ok=True)
            with open(file_input, "r") as f:
                all_input = json.load(f)
                self.all_texts = [input["text"] for input in all_input]
                ## fixme: mengyao_debug locomo quick fix
                if "locomo" in dataset_name :
                    self.all_texts = [f" {text}\n" if not text.startswith(" ") else text for text in self.all_texts]
                ##add \n in the end.
                self.all_texts = [f"{text}\n" if not text.endswith("\n") else text for text in self.all_texts]
            if os.path.exists(self.similar_index_file_path):
                self.similar_idx = np.load(self.similar_index_file_path)
                print(f"index load from {self.similar_index_file_path}")
            else:
                rerank_model = RerankModel(bge_model_path=preprocess_model_path)
                self.similar_idx = rerank_model.preprocess_build_faiss_index(
                    all_documents=self.all_texts,
                    topk=10
                )
                np.save(self.similar_index_file_path, self.similar_idx)
                rerank_model.clean_()
                del rerank_model
        if model_path != "":
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            config._attn_implementation = "sdpa"
            print(f"Loading {model_type} model...")
            if use_multi_gpu:
                print("Using multi-GPU with device_map='auto'")
            self.model, self.device_map = self.load_model(model_type, model_path, config, device, use_multi_gpu, max_memory)
        if draft_model_path != "":
            self.use_local_draft_model = use_local_draft_model
            if use_local_draft_model:
                print(f"Initialize draft model from {draft_model_path}...")
                draft_config = AutoConfig.from_pretrained(draft_model_path, trust_remote_code=True)
                print(f"draft_config={draft_config}")
                draft_config._attn_implementation = "sdpa"
                self.draft_model, _ = self.load_model(draft_model_type, draft_model_path, draft_config, draft_model_device, use_multi_gpu=False)
                self.draft_model.eval()
                self.draft_model_device=draft_model_device
                self.draft_model_url = ""
            else:
                print(f"using remote draftmodel.")
                assert draft_model_url != "" "draft_model_url must not be empty."
                self.draft_model_url = draft_model_url
                self.draft_model = None
                self.draft_model_device = None
            self.draft_model_tokenizer = AutoTokenizer.from_pretrained(draft_model_path, trust_remote_code=True)
            self.draft_model_tokenizers = []
            for i in range(16):
                self.draft_model_tokenizers.append(AutoTokenizer.from_pretrained(draft_model_path, trust_remote_code=True))
        else:
            self.use_local_draft_model = True
            print(f"Skipping draft model.")
            self.draft_model = None
            self.draft_model_device = ""

        if use_multi_gpu:
            self.input_device = "cuda:0"  # First GPU for inputs
            self.draft_model_input_device = "cuda:0"  # First GPU for inputs
        else:
            self.input_device = device
            self.draft_model_input_device = draft_model_device


    def sort_docs(self, retrieved_docs: list[str]):
        """
        对对话文本进行排序，格式: "Conversation Time: 2:32 pm on 29 January, 2023. ..."
        提取时间部分进行排序，如果有不符合格式的文档直接返回原列表
        """
        if not retrieved_docs:
            return retrieved_docs

        # 检查所有文档是否都符合对话时间格式
        def is_conversation_format(doc: str) -> bool:
            if not doc.startswith("Conversation Time: "):
                return False

            # 检查是否有时间部分和句点
            time_part_end = doc.find('.', len("Conversation Time: "))
            if time_part_end == -1:
                return False

            # 提取时间字符串
            time_str = doc[len("Conversation Time: "):time_part_end].strip()

            # 尝试解析时间
            try:
                time_part, date_part = time_str.split(" on ")
                datetime.strptime(time_part, "%I:%M %p")
                datetime.strptime(date_part, "%d %B, %Y")
                return True
            except (ValueError, AttributeError, IndexError):
                return False

        # 如果有任何一个文档不符合格式，直接返回
        if not all(is_conversation_format(doc) for doc in retrieved_docs):
            print(f"fail to sort doc!")
            return retrieved_docs

        # 从文档中提取时间并解析为datetime对象
        def parse_conversation_time(doc: str) -> datetime:
            # 找到第一个句点的位置
            dot_index = doc.find('.', len("Conversation Time: "))
            # 提取时间字符串
            time_str = doc[len("Conversation Time: "):dot_index].strip()

            # 解析时间
            time_part, date_part = time_str.split(" on ")
            time_obj = datetime.strptime(time_part, "%I:%M %p")
            date_obj = datetime.strptime(date_part, "%d %B, %Y")

            return datetime.combine(date_obj.date(), time_obj.time())

        # 排序并返回
        return sorted(retrieved_docs, key=parse_conversation_time)

    def encode_str(self,
                    text: str):
        prompt_tokens = self.draft_model_tokenizer.encode(text, add_special_tokens=False)
        return prompt_tokens

    def draft_one_question(self,
                           system_prompt: str,
                           passages: list[str],
                           query: str,
                           rate: float,
                           keyword: str="",
                           reverse_attn=False,
                           use_entropy_and_relevance=False,
                           must_choose_docs: list[str] = None,
                           use_weighted_diff_attention=False,
                           preprocess=False,
                           weighted_use_value=False,
                           weighted_use_kv=False,
                           must_choose_token_indices=[],
                           ):
        print(f"draft_one_question query={query}")
        compare_sim = None
        query_states = None
        mean_attn_weights = None
        recompute_tokens, recompute_tokens_list, sorted_index, sorted_index_before_resort, passages, selected_indices = find_all_substr_needs_recompute(
            draft_model=self.draft_model,
            draft_model_device=self.draft_model_device,
            tokenizer=self.draft_model_tokenizer,
            system_prompt=system_prompt,
            passages=passages,
            query=query,
            rate=rate,
            must_choose_token_indices=must_choose_token_indices,
            reverse_attn=reverse_attn,
            use_local_draft_model=self.use_local_draft_model,
            draft_model_url=self.draft_model_url,
            compare_sim=compare_sim,
            keyword=keyword,
            weighted_use_value=weighted_use_value,
            weighted_use_kv=weighted_use_kv,
            tokenizers=self.draft_model_tokenizers
        )
        torch.cuda.empty_cache()
        return recompute_tokens, recompute_tokens_list, passages, rate, sorted_index, sorted_index_before_resort, selected_indices


def generate_response_with_fusionrag(
        system_prompt: str,
        prefix: str,
        fusionrag_cache_list: list[str],
        query_prompt: str,
        fusion_rag_model,
        model: str = "qwen3-8b",
        max_tokens=5000,
        recomputation_rate=0.3,
        sglang_url="http://127.0.0.1:30003/v1/completions",
        sglang_url_prefiller="http://127.0.0.1:30003/v1/completions",
        must_choose_docs = None,
) -> (str, dict, int):

    if "kimi" in model.lower():
        template = {
            "DEFAULT_SYSTEM_PROMPT": f"<|im_system|>system<|im_middle|>\n{system_prompt}\n{prefix}",
            "USER_PROMPT": f"<|im_end|><|im_user|>user<|im_middle|>{query_prompt}<|im_end|><|im_assistant|>assistant<|im_middle|><think></think>"
        }
    elif "qwen" in model.lower():
        template = {
            "DEFAULT_SYSTEM_PROMPT": f"""<|im_start|>system\n{system_prompt}\n{prefix}""",
            "USER_PROMPT": f"""<|im_end|>\n<|im_start|>user\n\nQuestion: /no_think {query_prompt}<|im_end|>\n<|im_start|>assistant\nAnswer: </think>"""
        }
    elif "glm" in model.lower():
        template = {
            "DEFAULT_SYSTEM_PROMPT": f"""[gMASK]<sop><|system|>\n{system_prompt}\n{prefix}""",
            "USER_PROMPT": f"""<|user|>\n{query_prompt}/nothink<|assistant|>\n<think></think>"""
        }

    time_start = time.time()
    recompute_prefix_len = len(fusion_rag_model.encode_str(template["DEFAULT_SYSTEM_PROMPT"]))
    # recompute_raw_text_len = len(fusion_rag_model.encode_str(raw_text))

    must_choose_token_indices = []
    if must_choose_docs is not None:
        total_doc_token_len = len(fusion_rag_model.draft_model_tokenizer.encode(template["DEFAULT_SYSTEM_PROMPT"]))
        print(f"system len={total_doc_token_len}")
        for doc_idx, doc in enumerate(fusionrag_cache_list):
            doc_len = len(fusion_rag_model.draft_model_tokenizer.encode(doc))
            if doc_idx in must_choose_docs:
                must_choose_token_indices.extend([i for i in range(total_doc_token_len, total_doc_token_len + doc_len)])
            total_doc_token_len += doc_len


    recompute_tokens, recompute_tokens_list, retrieved_docs, recompute_rate, sorted_doc_index, sorted_doc_index_before, selected_indices = fusion_rag_model.draft_one_question(
        template["DEFAULT_SYSTEM_PROMPT"],  ## DEFAULT_SYSTEM_PROMPT
        fusionrag_cache_list,
        template["USER_PROMPT"],
        recomputation_rate,
        "",
        reverse_attn=False,
        use_entropy_and_relevance=False,
        must_choose_docs=[],
        use_weighted_diff_attention=False,
        preprocess=False,  ## if do preprocess
        weighted_use_value=False,
        weighted_use_kv=True,
        must_choose_token_indices=must_choose_token_indices
    )
    print(f"draft_one_question time = {time.time() - time_start}")

    try:
        time_start = time.time()
        content, usage, top_logprobs, real_recomputation_rate = run_one_question_sglang(
            DEFAULT_SYSTEM_PROMPT=template["DEFAULT_SYSTEM_PROMPT"],
            USER_PROMPT=template["USER_PROMPT"],
            MODEL=model,
            retrived_docs=fusionrag_cache_list,
            max_tokens=max_tokens,  ## max tokens.
            retrived_docs_relevant_docs=[],
            recompute_tokens=recompute_tokens,
            recompute_tokens_list=recompute_tokens_list,
            max_workers=1,  ## max_workers.
            recomputation_rate=recomputation_rate,
            model_use=model,
            endpoint_url=sglang_url,
            prefiller_endpoint_url=sglang_url_prefiller,
            method_keyword="",
        )
        print(f"run_one_question_sglang time: {time.time() - time_start} content={content}")

        usage_info = {
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
        }

        return content, usage_info, recompute_prefix_len+len(selected_indices)
    except Exception as e:
        print(e)