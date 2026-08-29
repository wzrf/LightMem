#!/usr/bin/env python
# coding=utf-8
'''
Description  :
Author       : Boxin Zhang, Azure-Tang
Version      : 0.1.0
Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
'''
import copy
from typing import Dict, Union, Tuple, List
import torch
from torch import nn
import itertools
import time
import enum
import re
import os
import math
import string
import gc
import json
import collections
import numpy as np
import gc
import requests
from transformers import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    MinPLogitsWarper,
    TypicalLogitsWarper,
    EpsilonLogitsWarper,
    EtaLogitsWarper,
    GenerationConfig,
    AutoTokenizer
)
from rouge import Rouge
from filelock import FileLock
from typing import List, Dict, Any, Optional
from collections import Counter

def find_connected_components(positions, max_gap=2, within=False):
    """
    找到位置列表中的连通分量（相邻 token 群组）

    Args:
        positions: 位置列表
        max_gap: 最大允许的间隔，小于等于这个间隔的位置被认为是连通的

    Returns:
        List of lists, 每个子列表是一个连通分量
    """
    if len(positions) == 0:
        return []

    positions = sorted(positions)
    components = []
    current_component = [positions[0]]
    connect_positions = set()

    for i in range(1, len(positions)):
        if positions[i] - positions[i-1] <= max_gap:
            if not within:
                current_component.append(positions[i])
            else:
                current_component.extend([p for p in range(positions[i - 1] + 1, positions[i] + 1)])
                for c in range(positions[i - 1] + 1, positions[i]):
                    connect_positions.add(c)
        else:
            components.append(current_component)
            current_component = [positions[i]]

    components.append(current_component)
    return components, list(connect_positions)

def smart_query_selection(attention_scores, doc_len, target_ratio, system_len, device='cpu', smarter=False,
                          eigenvalue=None, tokenizer=None, input_tokens=None, similarity=0.0, keyword=""):
    """
    Smart Query Selection: 使用连通性分析确保相关 token 群组被完整选中

    Args:
        attention_scores: torch.Tensor, shape [doc_len], 每个位置的 attention 分数
        doc_len: 文档长度
        target_ratio: 目标选择比例
        system_len: system prompt 长度
        device: 计算设备

    Returns:
        List of selected positions (global indices, including system_len offset)
    """
    if isinstance(attention_scores, torch.Tensor):
        attention_scores = attention_scores.float().cpu().numpy()

    target_count = int(doc_len * target_ratio)

    # Step 1: 找到高 attention 位置
    mean_attn = np.mean(attention_scores)
    std_attn = np.std(attention_scores)
    # threshold = mean_attn + 0.25 * std_attn ## 1/4 std
    threshold = mean_attn

    high_attn_positions = list(np.where(attention_scores > threshold)[0])
    # print(f"high_attn_positions = {high_attn_positions}")
    if eigenvalue is not None:
        eigenvalue["attention_scores"] = [float(t.item()) for t in attention_scores]

    # 1. max_gap=5, min_len=3
    # 2. max_gap=20, min_len=20
    # 2. max_gap=min_len=5, 0.5 * std_attn, min_chosen_weight = 0.4/0.2: this is the current best, but will leftout some important infos
    # 3. max_gap=min_len=5, 0.1 * std_attn, min_chosen_weight = 0.2: let more relevant data be found, but cut the irelevant
    # 4. max_gap=min_len=5, 0.25 * std_attn, min_chosen_weight = 0.0002
    # 5. max_gap=min_len=5, 0.25 * std_attn, min_chosen_weight = 0.01
    # Step 2: 连通分量分析
    connect_positions = []
    if smarter:
        max_gap = 5
        min_len = max_gap
        min_chosen_weight = 0.2
        if eigenvalue is not None:
            eigenvalue["max_gap"] = max_gap
            eigenvalue["min_len"] = min_len
            eigenvalue["min_chosen_weight"] = min_chosen_weight
        components, connect_positions = find_connected_components(high_attn_positions, max_gap=max_gap, within=True)
    else:
        max_gap = 5 ## mengyao_debug I changed this.
        if "no_smart_connect" in keyword: ## 不用连通性分析
            max_gap = 0
        components, _ = find_connected_components(high_attn_positions, max_gap=max_gap, within=False)


    # Step 3: 计算每个分量的总 attention
    component_scores = []
    for comp in components:
        total_score = sum(attention_scores[p] for p in comp if p not in connect_positions)
        component_scores.append((comp, total_score))

    # Step 4: 按总 attention 排序
    component_scores.sort(key=lambda x: x[1], reverse=True)
    component_scores_ = copy.deepcopy(component_scores)

    if smarter:
        max_score = 1e-8
        all_tokens = []
        if tokenizer is not None:
            input_tokens_ = torch.cat(input_tokens)
            for cs, score in component_scores:
                if len(cs) > min_len:
                    if max_score == 1e-8:
                        max_score = score # set max score
                input_str = tokenizer.decode(input_tokens_[cs], skip_special_tokens=True)
                prefix = ""
                if len(cs) >= min_len and score/max_score > min_chosen_weight:
                    prefix = "【Chosen】"
                print(f"{prefix} \033[31m{input_str}\033[0m, score={score} len={len(cs)}")
                #for debug
                chosen = False
                if len(cs) >= min_len and score/max_score > min_chosen_weight:
                    chosen = True
                all_tokens.append({
                    "str": input_str,
                    "score": float(score),
                    "len": len(cs),
                    "chosen": chosen
                })
        eigenvalue["chosen_tokens"] = all_tokens
        # mengyao_debug: make sure the token we select is not a single token and has some weights on it.
        component_scores = [cs for cs in component_scores if len(cs[0]) >= min_len and cs[1]/max_score > min_chosen_weight]


    # for cs in component_scores:
    #     print(f"component len={len(cs[0])}, score={cs[1]}")
    if eigenvalue != None:
        eigenvalue["components"] = len(component_scores)

    # Step 5: 贪心选择分量 + 上下文扩展 (±1)
    selected = set()
    selected_reserved = set()

    for comp, total_score in component_scores:
        # 扩展分量边界 (±1)
        extended_comp = set()
        for p in comp:
            if "no_smart_connect" in keyword:
                extended_comp.add(p)
            else:
                for offset in range(-1, 2):
                    new_p = p + offset
                    if 0 <= new_p < doc_len:
                        extended_comp.add(new_p)

        # 检查是否会超过目标 (允许 10% 余量)
        new_positions = extended_comp - selected
        if len(selected) + len(new_positions) <= target_count * 1.1:
            selected.update(extended_comp)

    for cs in component_scores_:
        if cs not in component_scores:
            comp, total_score = cs
            extended_comp = set()
            for p in comp:
                extended_comp.add(p)
            selected_reserved.update(extended_comp)

    # Step 6: 补充到目标数量
    if not smarter:
        if len(selected) < target_count:
            sorted_indices = np.argsort(attention_scores)[::-1]
            for pos in sorted_indices:
                if pos not in selected:
                    selected.add(int(pos))
                    if len(selected) >= target_count:
                        break

    # Step 7: 如果超过目标，移除最低分的位置
    while len(selected) > target_count:
        min_pos = min(selected, key=lambda p: attention_scores[p])
        selected.remove(min_pos)

    # 转换为全局索引 (加上 system_len 偏移)
    selected_global = [p + system_len for p in sorted(selected)]
    selected_global_reserved = [p + system_len for p in sorted(selected_reserved) if p not in connect_positions]
    # selected_global_reserved = [p + system_len for p in sorted(selected_reserved)]

    if smarter:
        return selected_global, selected_global_reserved

    return selected_global

def run_draft(prompt: str, prompt_list: list[str], draft_model_url:str):
    # API端点
    # 请求头
    headers = {
        "Content-Type": "application/json"
    }

    # 请求数据
    data = {
        "prompt": prompt,
        "max_tokens": 1,
        "fusionrag_params": {
            "prefix_prompt": "",
            "prompt_list": prompt_list,
        },
        "temperature": 0.0
    }

    try:
        # 发送POST请求
        response = requests.post(draft_model_url, headers=headers, json=data)
        result = response.json()
        # 检查响应状态
        if response.status_code == 200:
            print(f"draft请求成功")
        else:
            print(f"请求失败，状态码: {response.status_code}")
        return result
    except Exception as e:
        print(e)
        return {}

def call_remote_draft_model(system_prompt: str, docs: list[str], user_prompt: str, draft_model_url: str):
    prompt_list = [system_prompt]
    prompt_list.extend(docs)
    prompt_list.append(user_prompt)

    return run_draft(
        prompt="".join(prompt_list),
        prompt_list=prompt_list,
        draft_model_url=draft_model_url
    )

def highlight_tokens_compare(
        k_need_index: List[int],
        passages: Union[List[torch.Tensor], torch.Tensor],
        tokenizer,
        query: str = "",
        passages_str: List[str] = None
) -> Tuple[List[str], List[List[str]]]:
    """
    根据 passages_str 挨个 encode，计算出每个 Passage 在全局 Token 中的范围，
    然后使用 k_need_index 提取每个 Passage 内部的重算字符串列表 (recompute_str_list)。

    返回:
        combine_tokens: List[str] - 所有 passage 片段展平后的列表
        all_recompute_tokens: List[List[str]] - 与 passages_str 1对1对应的重算 str list
                                                 [0]不重算, [1]重算, [2]不重算...
    """
    if passages_str is None:
        raise ValueError("passages_str 不能为 None，每个 passage 需要通过 passages_str 来对齐！")

    k_need_set = set(k_need_index)
    all_recompute_tokens: List[List[str]] = []
    combine_tokens: List[str] = []

    global_token_offset = 0  # 记录当前 passage 在全局 full_passage 中的起始 token 偏移

    # 1. 遍历每一个文档字符串，单独 encode 找到各自的 token 边界
    for p_idx, p_str in enumerate(passages_str):
        # 对当前 passage 进行 encode，得到对应的 token ids
        p_tokens = tokenizer.encode(p_str, add_special_tokens=False)
        print(f"passage {p_idx}: {len(p_tokens)}")
        p_len = len(p_tokens)

        if p_len == 0:
            all_recompute_tokens.append([])
            continue

        combined_passages: List[List[int]] = []
        last_chosen = False  # 契约：首个片段必须是“不需要重算”的 (偶数索引)
        last_tokens: List[int] = []

        # 2. 对当前 Passage 内的 Token 逐个匹配全局 k_need_index
        for local_i, token_id in enumerate(p_tokens):
            global_i = global_token_offset + local_i
            is_needed = global_i in k_need_set

            if is_needed == last_chosen:
                last_tokens.append(int(token_id))
            else:
                last_chosen = is_needed
                # 当 local_i=0 且第一个 Token 就需要重算(is_needed=True)时，
                # 此处 last_tokens 为 []，append([]) 会在索引 0 放空 Token，
                # 解码为 ""，顺延高亮块到索引 1 (奇数位)，完美满足契约！
                combined_passages.append(last_tokens)
                last_tokens = [int(token_id)]

        if last_tokens:
            combined_passages.append(last_tokens)

        # 3. 在当前 Passage 内部进行增量前缀 Decode，生成该 Passage 的 recompute_str_list
        p_recompute_list: List[str] = []
        for i in range(len(combined_passages)):
            previous_text_combine = sum(combined_passages[:i], [])
            cur_text_combine = sum(combined_passages[:i + 1], [])

            previous_text = tokenizer.decode(previous_text_combine, skip_special_tokens=False)
            cur_text = tokenizer.decode(cur_text_combine, skip_special_tokens=False)

            p_recompute_list.append(cur_text[len(previous_text):])

        all_recompute_tokens.append(p_recompute_list)
        combine_tokens.extend(p_recompute_list)

        # 累加 Token 偏移量
        global_token_offset += p_len

    # 4. 终端彩色高亮打印（保持调试可视化）
    if isinstance(passages, list):
        full_passage = torch.cat(passages).squeeze()
    else:
        full_passage = passages

    full_passage_tokens = full_passage.tolist() if isinstance(full_passage, torch.Tensor) else full_passage
    tokens = [tokenizer.decode(t, skip_special_tokens=False) for t in full_passage_tokens]

    highlighted_tokens = []
    for i, token_str in enumerate(tokens):
        if i in k_need_set:
            highlighted_tokens.append(f"\033[1;31m{token_str}\033[0m")
        else:
            highlighted_tokens.append(token_str)

    highlighted_with_spaces = "".join(highlighted_tokens)

    if len(k_need_index) > 0:
        print(f"query={query}\n")
        print(f"highlighted_with_spaces={highlighted_with_spaces}")

    return combine_tokens, all_recompute_tokens

def find_all_substr_needs_recompute(draft_model, draft_model_device, tokenizer, system_prompt: str,
                                    passages: list[str], query: str, rate: float, must_choose_token_indices: list[int],
                                    weighted_use_value: bool, weighted_use_kv:bool, reverse_attn=False,
                                    use_local_draft_model=True, draft_model_url="", save_attention_heatmap=False, compare_sim=None, keyword="", tokenizers=None):
    system_prompt_tokens = tokenizer.encode(system_prompt, add_special_tokens = False)
    passages_with_system_prompt_str_list = [system_prompt]
    passages_with_system_prompt_str_list.extend(passages)
    # passages_full = "".join(passages)
    # passages_tokens = tokenizer.encode(passages_full, add_special_tokens=False)
    first_passage_token = tokenizer.encode(passages[0], add_special_tokens=False)
    each_passages_tokens = []
    passages_tokens = []
    sorted_index = []
    sorted_index_before_resort = []
    new_passages = copy.deepcopy(passages)
    for passage in passages:
        each_passages_tokens.append(tokenizer.encode(passage, add_special_tokens = False))
        passages_tokens.extend(each_passages_tokens[-1])

    query_tokens = tokenizer.encode(query, add_special_tokens = False)
    full_input = system_prompt_tokens + passages_tokens + query_tokens
    full_input_without_query = system_prompt_tokens + passages_tokens
    full_input_tensor = torch.tensor(full_input).unsqueeze(0).to(draft_model_device)

    result = call_remote_draft_model(
        system_prompt=system_prompt,
        docs=passages,
        user_prompt=query,
        draft_model_url=draft_model_url
    )
    attn_weights = result["choices"][0]["attention_weights"]
    attn_weights = attn_weights[len(system_prompt_tokens):]
    multi_layer_attn = torch.tensor(attn_weights)

    match = re.search(r'(?<=attention_weight_adjust_)([-+]?\d*\.\d+|\d+)', keyword)
    weight = float(match.group(1)) if match else 1.0
    print(f"attention weight={weight}")


    if reverse_attn:
        selected_indices = smart_query_selection(
            attention_scores=multi_layer_attn,
            doc_len=len(query_tokens),
            target_ratio=rate,
            system_len=len(system_prompt_tokens + passages_tokens),
            device=draft_model_device,
            keyword=keyword
        )
    else:
        selected_indices = smart_query_selection(
            attention_scores=multi_layer_attn,
            doc_len=len(passages_tokens),
            target_ratio=rate,
            system_len=len(system_prompt_tokens),
            device=draft_model_device,
            keyword=keyword
        )

    selected_indices.extend(must_choose_token_indices)
    selected_indices = sorted(list(set(selected_indices)))
    time_start = time.time()
    if reverse_attn:
        combine_tokens, all_recompute_tokens = highlight_tokens_compare(selected_indices, torch.tensor(full_input), tokenizer, query=query,
                                    passages_str=passages_with_system_prompt_str_list)
    else:
        combine_tokens, all_recompute_tokens = highlight_tokens_compare(selected_indices, torch.tensor(full_input_without_query), tokenizer, query=query,
                                    passages_str=passages_with_system_prompt_str_list)
    print(f"highlight_tokens_compare takes time = {time.time() - time_start}")

    return combine_tokens, all_recompute_tokens, sorted_index, sorted_index_before_resort, new_passages, selected_indices

