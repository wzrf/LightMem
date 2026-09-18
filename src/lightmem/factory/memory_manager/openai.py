import concurrent
from collections import defaultdict
from openai import OpenAI
from typing import List, Dict, Optional, Literal, Any
import json, os, warnings
import httpx
from lightmem.memory.prompts import EXTRACTION_PROMPTS, METADATA_GENERATE_PROMPT
from lightmem.configs.memory_manager.base_config import BaseMemoryManagerConfig
from lightmem.memory.utils import clean_response, clean_json
from lightmem.fusionrag.run_question import FusionRAGModel
from lightmem.fusionrag.sglang_kvcache import run_one_question_sglang

model_name_context_windows = {
    "gpt-4o-mini": 128000,
    "qwen3-30b-a3b-instruct-2507": 128000,
    "glm-4.6": 200000,
    "DEFAULT": 128000,  # Recommended default context window
}


class OpenaiManager:
    def __init__(self, config: BaseMemoryManagerConfig):
        self.config = config

        self.recomputation_rate = 0.5
        self.sglang_url = self.config.openai_base_url + "/completions"
        self.sglang_url_prefiller = self.config.openai_base_url + "/completions"
        if os.getenv("FUSIONRAG", "").lower() == "true":
            self.fusion_rag_model = FusionRAGModel(
                model_path='',
                use_multi_gpu=True,
                model_type="qwen3",
                model_name="Qwen3-32B",
                draft_model_type="qwen",
                draft_model_name="qwen2.5-3b",
                preprocess_model_path="/data2/qy_tmp/xumengyao/bge-m3",
                draft_model_path="/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct",
                draft_model_url="http://192.168.0.238:30005/v1/completions",
                apikey="xxx",
                use_local_draft_model=False,
            )
        else:
            self.fusion_rag_model = None

        if not self.config.model:
            self.config.model = "gpt-4o-mini"
        
        if self.config.model in model_name_context_windows:
            self.context_windows = model_name_context_windows[self.config.model]
        else:
            self.context_windows = model_name_context_windows["DEFAULT"]

        http_client = httpx.Client(verify=False)

        if os.environ.get("OPENROUTER_API_KEY"):  # Use OpenRouter
            self.client = OpenAI(
                api_key=os.environ.get("OPENROUTER_API_KEY"),
                base_url=self.config.openrouter_base_url
                or os.getenv("OPENROUTER_API_BASE")
                or "https://openrouter.ai/api/v1",
            )
        else:
            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            base_url = (
                self.config.openai_base_url
                or os.getenv("OPENAI_API_BASE")
                or os.getenv("OPENAI_BASE_URL")
                or "https://api.openai.com/v1"
            )

            self.client = OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)

    def _parse_response(self, response, tools):
        """
        Process the response based on whether tools are used or not.

        Args:
            response: The raw response from API.
            tools: The list of tools provided in the request.

        Returns:
            str or dict: The processed response.
        """
        if tools:
            processed_response = {
                "content": response.choices[0].message.content,
                "tool_calls": [],
            }

            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    processed_response["tool_calls"].append(
                        {
                            "name": tool_call.function.name,
                            "arguments": json.loads(tool_call.function.arguments),
                        }
                    )

            return processed_response
        else:
            return response.choices[0].message.content


    def generate_response_with_fusionrag(
        self,
        system_prompt: str,
        prefix: str,
        fusionrag_cache_list: list[str],
        query_prompt: str,
        model: str="qwen3-8b",
        max_tokens = 5000
    ) -> (str, dict, dict):

        template = {
            "DEFAULT_SYSTEM_PROMPT": f"""<|im_start|>system\n{system_prompt}\n{prefix}""",
            "USER_PROMPT": f"""<|im_end|>\n<|im_start|>user\n\nQuestion: /no_think {query_prompt}<|im_end|>\n<|im_start|>assistant\nAnswer: </think>"""
        }

        fusionrag_cache_list_text = "".join(fusionrag_cache_list)
        system_len = len(self.fusion_rag_model.draft_model_tokenizer.encode(template["DEFAULT_SYSTEM_PROMPT"]))
        query_len = len(self.fusion_rag_model.draft_model_tokenizer.encode(template["USER_PROMPT"]))
        origin_text_list_len = len(self.fusion_rag_model.draft_model_tokenizer.encode(fusionrag_cache_list_text))

        recompute_tokens, recompute_tokens_list, retrieved_docs, recompute_rate, sorted_doc_index, sorted_doc_index_before, selected_indices = self.fusion_rag_model.draft_one_question(
            template["DEFAULT_SYSTEM_PROMPT"],  ## DEFAULT_SYSTEM_PROMPT
            fusionrag_cache_list,
            template["USER_PROMPT"],
            self.recomputation_rate,
            "",
            False,
            False,
            [],
            False,
            False,  ## if do preprocess
            False,
            True
        )

        try:
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
                recomputation_rate=self.recomputation_rate,
                model_use=model,
                endpoint_url=self.sglang_url,
                prefiller_endpoint_url=self.sglang_url_prefiller,
                method_keyword="",
            )

            usage_info = {
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
            }

            return content, usage_info, {
                "system_len": system_len,
                "query_len": query_len,
                "origin_text_list_len": origin_text_list_len,
                "fusionrag_text_list_len":  len(selected_indices),
            }
        except Exception as e:
            print(e)


    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, str]] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
    ) -> Optional[str]:
        """
        Generate a response based on the given messages.

        Args:
            messages (list): List of message dicts containing 'role' and 'content'.
            response_format (str or object, optional): Format of the response. Defaults to "text".
            tools (list, optional): List of tools that the model can call. Defaults to None.
            tool_choice (str, optional): Tool choice method. Defaults to "auto".

        Returns:
            str: The generated response.
        """
        params = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
        }

        if os.getenv("OPENROUTER_API_KEY"):
            openrouter_params = {}
            
            models = getattr(self.config, 'models', None)    
            route = getattr(self.config, 'route', 'fallback') 
            if models:
                openrouter_params["models"] = models
                openrouter_params["route"] = route
                params.pop("model")

            if self.config.site_url and self.config.app_name:
                extra_headers = {
                    "HTTP-Referer": self.config.site_url,
                    "X-Title": self.config.app_name,
                }
                openrouter_params["extra_headers"] = extra_headers

            params.update(**openrouter_params)

        ##mengyao_debug for sglang
        # if response_format:
        #     params["response_format"] = response_format
        if tools:  # TODO: Remove tools if no issues found with new memory addition logic
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        ##mengyao_debug
        params["extra_body"] = {
            "chat_template_kwargs":
                {
                    "thinking": False,
                    "enable_thinking": False
                }
        }

        response = self.client.chat.completions.create(**params)
        usage_info = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        parsed_response = self._parse_response(response, tools)

        return parsed_response, usage_info

    def meta_text_extract(
        self,
        extract_list: List[List[List[Dict]]],
        messages_use: Literal["user_only", "assistant_only", "hybrid"] = "user_only",
        topic_id_mapping: Optional[List[List[int]]] = None,
        extraction_mode: Literal["flat", "event"] = "flat",
        custom_prompts: Optional[Dict[str, str]] = None  
    ) -> List[Optional[Dict]]:
        """
        Extract metadata from text segments using parallel processing.

        Args:
            extract_list: List of message segments to process
            messages_use: Strategy for which messages to use
            topic_id_mapping: For each API call, the global topic IDs
            extraction_mode: "flat" or "event"
            custom_prompts: Optional custom prompts. If None, use defaults from EXTRACTION_PROMPTS

        Returns:
            List of extracted metadata results, None for failed segments
        """
        if not extract_list:
            return []
        
        default_prompts = EXTRACTION_PROMPTS.get(extraction_mode, {})
        
        if custom_prompts is None:
            prompts = default_prompts
        else:
            prompts = {**default_prompts, **custom_prompts}
        
        if extraction_mode == "flat":
            return self._extract_with_prompt(
                system_prompt=prompts.get("factual", METADATA_GENERATE_PROMPT),
                extract_list=extract_list,
                messages_use=messages_use,
                topic_id_mapping=topic_id_mapping,
                entry_type="factual"
            )
        
        elif extraction_mode == "event":
            factual_results = self._extract_with_prompt(
                system_prompt=prompts["factual"],
                extract_list=extract_list,
                messages_use=messages_use,
                topic_id_mapping=topic_id_mapping,
                entry_type="factual"
            )
            
            relational_results = self._extract_with_prompt(
                system_prompt=prompts["relational"],
                extract_list=extract_list,
                messages_use=messages_use,
                topic_id_mapping=topic_id_mapping,
                entry_type="relational",
            )
            
            return self._merge_dual_perspective_results(
                factual_results, 
                relational_results
            )
        
        else:
            raise ValueError(f"Unknown extraction_mode: {extraction_mode}")
    
    def _merge_dual_perspective_results(
        self,
        factual_results: List[Optional[Dict]],
        relational_results: List[Optional[Dict]]
    ) -> List[Optional[Dict]]:
        """
        Args:
            factual_results: Factual extraction results
            relational_results: Relational extraction results
        
        Returns:
            Merged results with combined cleaned_result and accumulated usage
        """
        merged_results = []
        
        for factual, relational in zip(factual_results, relational_results):
            if factual is None and relational is None:
                merged_results.append(None)
                continue
            
            merged = {
                "input_prompt": [],
                "output_prompt": "",
                "cleaned_result": [],
                "fusionrag_stats": [],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "reuse_history_tokens": 0,
                    "straight_use_tokens": 0,
                }
            }
            
            if factual is not None:
                merged["input_prompt"].extend(factual.get("input_prompt", []))
                merged["cleaned_result"].extend(factual.get("cleaned_result", []))
                fusionrag_stats = factual.get("fusionrag_stats", None)
                if fusionrag_stats is not None:
                    merged["fusionrag_stats"].extend(fusionrag_stats)
                if factual.get("usage"):
                    for key in merged["usage"]:
                        merged["usage"][key] += factual["usage"].get(key, 0)
            
            if relational is not None:
                merged["input_prompt"].extend(relational.get("input_prompt", []))
                merged["cleaned_result"].extend(relational.get("cleaned_result", []))
                fusionrag_stats = relational.get("fusionrag_stats", None)
                if fusionrag_stats is not None:
                    merged["fusionrag_stats"].extend(fusionrag_stats)
                if relational.get("usage"):
                    for key in merged["usage"]:
                        merged["usage"][key] += relational["usage"].get(key, 0)
            
            merged["output_prompt"] = (
                f"Factual: {factual.get('output_prompt', 'N/A') if factual else 'N/A'}\n"
                f"Relational: {relational.get('output_prompt', 'N/A') if relational else 'N/A'}"
            )
            
            merged_results.append(merged)
        
        return merged_results

    def _extract_with_prompt(
        self,
        system_prompt: str,
        extract_list: List[List[List[Dict]]],
        messages_use: str,
        topic_id_mapping: Optional[List[List[int]]],
        entry_type: str = "factual",
    ) -> List[Optional[Dict]]:
        """
        Args:
            system_prompt: System prompt for extraction
            extract_list: List of message segments
            messages_use: Message filtering strategy
            topic_id_mapping: Global topic IDs
            entry_type: "factual" or "relational"
        
        Returns:
            List of extraction results
        """
        def concatenate_messages(segment: List[Dict], messages_use: str) -> str:
            """Concatenate messages based on usage strategy"""
            role_filter = {
                "user_only": {"user"},
                "assistant_only": {"assistant"},
                "hybrid": {"user", "assistant"}
            }

            if messages_use not in role_filter:
                raise ValueError(f"Invalid messages_use value: {messages_use}")

            allowed_roles = role_filter[messages_use]
            message_lines = []

            for mes in segment:
                if mes.get("role") in allowed_roles:
                    sequence_id = mes["sequence_number"]
                    role = mes["role"]
                    content = mes.get("content", "")
                    speaker_name = mes.get("speaker_name", "")
                    time_stamp = mes.get("time_stamp", "")
                    weekday = mes.get("weekday", "")
                    
                    time_prefix = ""
                    if time_stamp and weekday:
                        time_prefix = f"[{time_stamp}, {weekday}] "

                    if speaker_name:
                        message_lines.append(f"{time_prefix}{sequence_id//2}.{speaker_name}: {content}")
                    else:
                        message_lines.append(f"{time_prefix}{sequence_id//2}.{role}: {content}")
            
            return "\n".join(message_lines)

        max_workers = min(len(extract_list), 5)

        def process_segment_wrapper(args):
            api_call_idx, api_call_segments = args
            try:
                user_prompt_parts: List[str] = []
                
                global_topic_ids: List[int] = []
                if topic_id_mapping and api_call_idx < len(topic_id_mapping):
                    global_topic_ids = topic_id_mapping[api_call_idx]

                for topic_idx, topic_segment in enumerate(api_call_segments):
                    if topic_idx < len(global_topic_ids):
                        global_topic_id = global_topic_ids[topic_idx]
                    else:
                        global_topic_id = topic_idx + 1
                    
                    topic_text = concatenate_messages(topic_segment, messages_use)
                    user_prompt_parts.append(f"--- Topic {global_topic_id} ---\n{topic_text}")

                print(f"User prompt for API call {api_call_idx}:\n" + "\n".join(user_prompt_parts))
                user_prompt = "\n".join(user_prompt_parts)
                
                metadata_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]

                if os.getenv("FUSIONRAG", "").lower() == "true":
                    raw_response, usage_info, fusionrag_stats = self.generate_response_with_fusionrag(
                        system_prompt=system_prompt,
                        prefix="",
                        fusionrag_cache_list=["Now here is the real conversation: " + user_prompt],
                        query_prompt="Now extract **all possible facts or information** about the speakers from the real conversation."
                    )
                    fusionrag_stats["reuse_type"] = "reuse_prefill"
                else:
                    fusionrag_stats = {}

                    raw_response, usage_info = self.generate_response(
                        messages=metadata_messages,
                        response_format={"type": "json_object"},
                    )
                metadata_facts = clean_response(raw_response)

                if entry_type == "factual":
                    usage_info["straight_use_tokens"] = len(user_prompt)
                elif entry_type == "relational":
                    usage_info["reuse_history_tokens"] = len(user_prompt)
                
                for entry in metadata_facts:
                    entry["entry_type"] = entry_type

                return {
                    "input_prompt": metadata_messages,
                    "output_prompt": raw_response,
                    "cleaned_result": metadata_facts,
                    "usage": usage_info,
                    "fusionrag_stats": [fusionrag_stats],
                    "entry_type": entry_type
                }
                
            except Exception as e:
                print(f"Error processing API call {api_call_idx}: {e}")
                return {
                    "input_prompt": [],
                    "output_prompt": "",
                    "cleaned_result": [],
                    "usage": None,
                    "fusionrag_stats": None,
                    "entry_type": entry_type
                }

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            try:
                results = list(executor.map(process_segment_wrapper, enumerate(extract_list)))
            except Exception as e:
                print(f"Error in parallel processing: {e}")
                results = [None] * len(extract_list)

        return results

    def _call_update_llm(self, system_prompt, target_entry, candidate_sources):
        target_memory = target_entry["payload"]["memory"]
        candidate_memories = [c["payload"]["memory"] for c in candidate_sources]

        user_prompt = (
            f"Target memory:{target_memory}\n"
            f"Candidate memories:\n" + "\n".join([f"- {m}" for m in candidate_memories])
        )

        user_prompt_list = [
            f"Target memory: {target_memory}",
            *[f"Candidate memories: {m}" for m in candidate_memories]
        ]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        if os.getenv("FUSIONRAG", "").lower() == "true":
            response_text, usage_info, fusionrag_stats = self.generate_response_with_fusionrag(
                system_prompt=system_prompt,
                prefix="",
                fusionrag_cache_list=user_prompt_list,
                query_prompt="Now decide whether the target memory should be updated, deleted, or ignored.",
            )
            fusionrag_stats["reuse_type"] = "reuse_decode"
        else:
            fusionrag_stats = {}

            response_text, usage_info = self.generate_response(
                messages=messages,
                response_format={"type": "json_object"}
            )

        response_text = clean_json(response_text)
        
        try:
            result = json.loads(response_text)
            if "action" not in result:
                result = {"action": "ignore"}
            result["usage"] = usage_info
            result["fusionrag_stats"] = fusionrag_stats
            return result
        except Exception:
            return {"action": "ignore", "usage": usage_info if 'usage_info' in locals() else None, "fusionrag_stats": fusionrag_stats}
