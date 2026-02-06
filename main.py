import io
import json
import sys
import gc
import re
from typing import Any, TypeVar
from dataclasses import dataclass
import base64
import asyncio
from browser_use import Agent, Browser
import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Qwen2VLImageProcessorFast,
    Qwen3VLForConditionalGeneration,
    Qwen3VLProcessor,
    Qwen3VLVideoProcessor,
    BitsAndBytesConfig,
)
from browser_use.llm.base import BaseChatModel
from browser_use.llm.messages import BaseMessage, ContentPartImageParam, ContentPartTextParam
from browser_use.llm.views import ChatInvokeCompletion
from pydantic import BaseModel, ValidationError
import os
# Must be set before importing torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


T = TypeVar("T", bound=BaseModel)


@dataclass
class Qwen3VLLocal(BaseChatModel):
    model: str = os.getenv("QWEN_VL_MODEL", "Qwen/Qwen3-VL-2B-Instruct")
    _model: Any | None = None
    _processor: Any | None = None
    _device: str | None = None

    @property
    def provider(self) -> str:
        return "qwen3-vl-local"

    @property
    def name(self) -> str:
        return self.model

    @property
    def model_name(self) -> str:
        return self.model

    def _select_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _load_config(self) -> Any | None:
        try:
            config = AutoConfig.from_pretrained(self.model)
        except Exception:
            return None
        if hasattr(config, "text_config"):
            text_config = config.text_config
            pad_token_id = getattr(text_config, "pad_token_id", None)
            if pad_token_id is None:
                pad_token_id = getattr(config, "pad_token_id", None)
            if pad_token_id is None:
                pad_token_id = getattr(config, "eos_token_id", None)
            if pad_token_id is None:
                pad_token_id = 151643
            text_config.pad_token_id = pad_token_id
        return config

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        self._device = self._select_device()
        print(f"🚀 Loading model on {self._device.upper()}...", flush=True)
        if self._device == "cuda":
            try:
                # Maximize GPU memory utilization
                torch.cuda.set_per_process_memory_fraction(0.98)
                torch.cuda.empty_cache()
            except Exception:
                pass
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        config = self._load_config()

        # Configure 4-bit quantization
        quantization_config = None
        if self._device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        device_map = "auto" if self._device == "cuda" else None
        # torch_dtype is handled by quantization_config if present, but good to set for non-quantized parts
        torch_dtype = torch.float16 if self._device == "cuda" else None

        load_kwargs: dict[str, Any] = {
            "device_map": device_map,
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
            # Use Scaled Dot Product Attention for memory efficiency
            "attn_implementation": "sdpa",
        }

        if quantization_config:
            load_kwargs["quantization_config"] = quantization_config

        if self._device == "cuda":
            try:
                # Limit memory to leave room for inference
                total_mem = torch.cuda.get_device_properties(0).total_memory
                # Reserve 5% for system/other apps, allow 95% for model
                max_memory: dict[Any, Any] = {0: int(total_mem * 0.90)}
                if any(x in self.model.upper() for x in ("4B", "7B", "8B", "9B", "14B")):
                    max_memory["cpu"] = "48GiB"
                load_kwargs["max_memory"] = max_memory
            except Exception:
                pass
        if config is not None:
            load_kwargs["config"] = config

        try:
            self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model, **load_kwargs)
        except Exception as e:
            print(f"Error loading model: {e}", flush=True)
            raise
        self._model.eval()

        try:
            self._processor = Qwen3VLProcessor.from_pretrained(self.model)
        except Exception:
            image_processor = Qwen2VLImageProcessorFast.from_pretrained(
                self.model)
            video_processor = Qwen3VLVideoProcessor.from_pretrained(self.model)
            tokenizer = AutoTokenizer.from_pretrained(self.model)
            self._processor = Qwen3VLProcessor(
                image_processor=image_processor,
                video_processor=video_processor,
                tokenizer=tokenizer,
            )

    def _image_from_data_url(self, data_url: str) -> Image.Image:
        encoded = data_url.split(",", 1)[1]
        data = base64.b64decode(encoded)
        return Image.open(io.BytesIO(data)).convert("RGB")

    def _content_parts_from_message(self, message: BaseMessage) -> list[dict[str, Any]]:
        if message.content is None:
            return []
        if isinstance(message.content, str):
            return [{"type": "text", "text": message.content}]
        parts: list[dict[str, Any]] = []
        if isinstance(message.content, list):
            for part in message.content:
                if isinstance(part, ContentPartTextParam):
                    parts.append({"type": "text", "text": part.text})
                elif isinstance(part, ContentPartImageParam):
                    url = part.image_url.url
                    if url.startswith("data:"):
                        image = self._image_from_data_url(url)
                        parts.append({"type": "image", "image": image})
                    else:
                        parts.append({"type": "image", "image": url})
        return parts

    def _resolve_schema_ref(self, node: Any, defs: dict[str, Any]) -> Any:
        if not isinstance(node, dict):
            return node
        ref = node.get("$ref")
        if not isinstance(ref, str):
            return node
        prefix = "#/$defs/"
        if ref.startswith(prefix):
            name = ref[len(prefix):]
            return defs.get(name, node)
        return node

    def _compact_structured_output_prompt(self, output_format: type[T]) -> str:
        schema = output_format.model_json_schema()
        defs = schema.get("$defs") if isinstance(schema, dict) else None
        if not isinstance(defs, dict):
            defs = {}

        action_keys: set[str] = set()
        try:
            action_schema = (schema.get("properties")
                             or {}).get("action") or {}
            items = action_schema.get("items") or {}
            items = self._resolve_schema_ref(items, defs)
            variants = items.get("anyOf") or items.get("oneOf") or []
            if isinstance(variants, list):
                for variant in variants:
                    resolved = self._resolve_schema_ref(variant, defs)
                    props = resolved.get("properties") if isinstance(
                        resolved, dict) else None
                    if isinstance(props, dict):
                        for k in props.keys():
                            if isinstance(k, str):
                                action_keys.add(k)
        except Exception:
            action_keys = set()

        allowed_keys = ", ".join(sorted(action_keys)) if action_keys else ""
        required = schema.get("required") if isinstance(schema, dict) else None
        required_keys = required if isinstance(required, list) else []
        required_keys = [k for k in required_keys if isinstance(k, str)]
        required_set = set(required_keys)
        top_props = schema.get("properties") if isinstance(
            schema, dict) else None
        top_props = top_props if isinstance(top_props, dict) else {}
        top_level_keys = set(top_props.keys())
        if "thinking" in top_level_keys:
            top_level_keys.add("thinking")
        if required_set:
            top_level_keys |= required_set
        ordered_top_keys = [k for k in required_keys if k in top_level_keys] + sorted(
            [k for k in top_level_keys if k not in required_set]
        )

        msg_parts: list[str] = [
            "Return ONLY valid JSON.\n",
            f"Top-level keys: {', '.join(ordered_top_keys) if ordered_top_keys else 'follow the provided schema'}.\n",
        ]
        if required_keys:
            msg_parts.append(f"Required keys: {', '.join(required_keys)}.\n")
        if "action" in top_level_keys:
            msg_parts.append("action must be a non-empty list.\n")
            msg_parts.append(
                "Each action item must be an object with EXACTLY ONE key (the action name) and its params.\n"
            )
        msg_parts.append("Do NOT include a 'type' field anywhere.\n")
        msg_parts.append(
            "Do NOT split action name and params into separate list items.\n")
        if "action" in top_level_keys:
            msg_parts.append(
                "You MUST include 'action' in the JSON object. Do not output only 'thinking'.\n"
            )
        if "thinking" in top_level_keys:
            msg_parts.append("Keep 'thinking' concise and to the point.\n")
        msg_parts.append(
            "IMPORTANT: For 'click' and 'input' actions, the 'index' field is MANDATORY.\n"
        )
        msg = "".join(msg_parts)
        if allowed_keys:
            msg += f"Allowed action keys: {allowed_keys}\n"
        example_lines: list[str] = [
            "Correct Example:\n{"] if top_level_keys else ["Correct Example:\n{"]
        for k in required_keys:
            if k == "action":
                continue
            example_lines.append(f'  "{k}": "...",')
        if "thinking" in top_level_keys and "thinking" not in required_set:
            example_lines.append('  "thinking": "...",')
        if "action" in top_level_keys:
            example_lines.append('  "action": [')
            example_lines.append('    {"click": {"index": 5}},')
            example_lines.append(
                '    {"input": {"index": 1, "text": "hello"}}')
            example_lines.append("  ]")
        else:
            if example_lines[-1].endswith(","):
                example_lines[-1] = example_lines[-1][:-1]
        example_lines.append("}")
        msg += "\n".join(example_lines)
        if "action" in top_level_keys:
            msg += (
                "\nIncorrect Example (DO NOT DO THIS):\n"
                "{\n"
                "  \"action\": [\"click\", {\"index\": 5}]\n"
                "}"
            )
        return msg

    def _serialize_messages(self, messages: list[BaseMessage], output_format: type[T] | None) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        if output_format is not None:
            msg = self._compact_structured_output_prompt(output_format)
            serialized.append(
                {"role": "system", "content": [
                    {"type": "text", "text": msg}]}
            )
        for message in messages:
            content = self._content_parts_from_message(message)
            serialized.append({"role": message.role, "content": content})
        return serialized

    def _clone_serialized_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cloned: list[dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            content = m.get("content")
            if isinstance(content, list):
                parts: list[Any] = []
                for part in content:
                    if isinstance(part, dict):
                        parts.append(dict(part))
                    else:
                        parts.append(part)
                cloned.append({"role": role, "content": parts})
            else:
                cloned.append(dict(m))
        return cloned

    def _truncate_text_parts(self, message: dict[str, Any], char_limit: int) -> None:
        content = message.get("content")
        if isinstance(content, str):
            if len(content) > char_limit:
                message["content"] = content[:char_limit] + "... [TRUNCATED]"
            return
        if not isinstance(content, list):
            return
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text", "")
                if isinstance(text, str) and len(text) > char_limit:
                    part["text"] = text[:char_limit] + "... [TRUNCATED]"

    def _downscale_images_in_messages(self, messages: list[dict[str, Any]], max_size: tuple[int, int]) -> bool:
        changed = False
        max_w, max_h = max_size
        for m in messages:
            content = m.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "image":
                    continue
                img = part.get("image")
                if not isinstance(img, Image.Image):
                    continue
                w, h = img.size
                if w <= max_w and h <= max_h:
                    continue
                ratio = min(max_w / max(w, 1), max_h / max(h, 1))
                new_w = max(1, int(w * ratio))
                new_h = max(1, int(h * ratio))
                part["image"] = img.resize((new_w, new_h), Image.BICUBIC)
                changed = True
        return changed

    def _generate(self, messages: list[dict[str, Any]], max_new_tokens: int) -> str:
        import time
        start_time = time.time()

        def tokenize(msgs: list[dict[str, Any]]) -> Any:
            return self._processor.apply_chat_template(
                msgs,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )

        def enforce_limits(
            original_messages: list[dict[str, Any]],
            max_input_tokens: int,
            char_limit: int,
            image_max_size: tuple[int, int],
        ) -> tuple[Any, list[dict[str, Any]]]:
            msgs = self._clone_serialized_messages(original_messages)
            inputs_local = tokenize(msgs)
            if inputs_local.input_ids.shape[1] <= max_input_tokens:
                return inputs_local, msgs

            print(
                f"WARNING: Input tokens ({inputs_local.input_ids.shape[1]}) exceed limit ({max_input_tokens}). Truncating history...",
                flush=True,
            )

            while inputs_local.input_ids.shape[1] > max_input_tokens and len(msgs) > 2:
                msgs.pop(1)
                inputs_local = tokenize(msgs)

            if inputs_local.input_ids.shape[1] > max_input_tokens:
                print(
                    f"WARNING: Single message too large ({inputs_local.input_ids.shape[1]}). Truncating text content...",
                    flush=True,
                )
                self._truncate_text_parts(msgs[-1], char_limit)
                inputs_local = tokenize(msgs)

            current_max_size = image_max_size
            while inputs_local.input_ids.shape[1] > max_input_tokens and current_max_size[0] > 256 and current_max_size[1] > 144:
                print(
                    f"WARNING: Still too large ({inputs_local.input_ids.shape[1]}). Downscaling images to {current_max_size[0]}x{current_max_size[1]}...",
                    flush=True,
                )
                changed = self._downscale_images_in_messages(
                    msgs, current_max_size)
                if not changed:
                    break
                inputs_local = tokenize(msgs)
                current_max_size = (
                    int(current_max_size[0] * 0.8), int(current_max_size[1] * 0.8))

            if inputs_local.input_ids.shape[1] > max_input_tokens:
                for k in ("input_ids", "attention_mask"):
                    t = getattr(inputs_local, k, None)
                    if t is not None and getattr(t, "dim", lambda: 0)() == 2 and t.shape[1] > max_input_tokens:
                        setattr(inputs_local, k, t[:, -max_input_tokens:])

            print(
                f"DEBUG: New input token count: {inputs_local.input_ids.shape[1]}", flush=True)
            return inputs_local, msgs

        if self._device == "cuda":
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                free_ratio = free_bytes / max(total_bytes, 1)
            except Exception:
                free_ratio = 0.0
        else:
            free_ratio = 0.0

        model_is_large = any(x in self.model.upper()
                             for x in ("4B", "7B", "8B", "9B", "14B"))

        default_max_input_tokens = 4096 if model_is_large else 10000
        try:
            max_input_tokens_target = int(
                os.getenv("QWEN_MAX_INPUT_TOKENS", str(default_max_input_tokens))
            )
        except Exception:
            max_input_tokens_target = default_max_input_tokens

        if free_ratio > 0.40:
            try:
                scale = float(os.getenv("QWEN_MAX_INPUT_TOKENS_FREE_SCALE", "1.15"))
            except Exception:
                scale = 1.15
            max_input_tokens_target = int(max_input_tokens_target * scale)

        default_char_limit = 12000 if model_is_large else 16000
        try:
            char_limit_target = int(os.getenv("QWEN_CHAR_LIMIT", str(default_char_limit)))
        except Exception:
            char_limit_target = default_char_limit

        image_max_size_target = (960, 540) if free_ratio > 0.35 else (768, 432)

        generated_ids = None
        inputs = None
        if self._device == "cuda":
            torch.cuda.empty_cache()
            print(
                f"DEBUG: VRAM allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB",
                flush=True,
            )
            print(
                f"DEBUG: VRAM reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB",
                flush=True,
            )

        try:
            attempt = 0
            current_max_new_tokens = min(max_new_tokens, 256)
            current_max_input_tokens = max_input_tokens_target
            current_char_limit = char_limit_target
            current_image_max = image_max_size_target
            use_cache = True

            always_no_cache = os.getenv("QWEN_ALWAYS_NO_CACHE", "").strip().lower() in (
                "1",
                "true",
                "yes",
                "y",
                "on",
            )
            try:
                disable_cache_over_tokens = int(
                    os.getenv("QWEN_DISABLE_CACHE_OVER_TOKENS", "9000")
                )
            except Exception:
                disable_cache_over_tokens = 9000
            try:
                max_new_tokens_no_cache = int(os.getenv("QWEN_MAX_NEW_TOKENS_NO_CACHE", "160"))
            except Exception:
                max_new_tokens_no_cache = 160

            while True:
                attempt += 1
                try:
                    inputs, _ = enforce_limits(
                        messages, current_max_input_tokens, current_char_limit, current_image_max)
                    inputs = inputs.to(self._model.device)
                    token_count = inputs.input_ids.shape[1]

                    if always_no_cache or token_count >= disable_cache_over_tokens:
                        use_cache = False
                        if current_max_new_tokens > max_new_tokens_no_cache:
                            current_max_new_tokens = max_new_tokens_no_cache

                    print(f"DEBUG: Input tokens: {token_count}", flush=True)
                    print(
                        f"DEBUG: Generating (max_new_tokens={current_max_new_tokens}, use_cache={use_cache})...",
                        flush=True,
                    )
                    with torch.inference_mode():
                        generated_ids = self._model.generate(
                            **inputs,
                            max_new_tokens=current_max_new_tokens,
                            do_sample=False,
                            num_beams=1,
                            repetition_penalty=1.1,
                            use_cache=use_cache,
                        )
                    break
                except RuntimeError as e:
                    if self._device != "cuda" or "out of memory" not in str(e).lower():
                        raise
                    if attempt >= 3:
                        raise
                    try:
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                    except Exception:
                        pass

                    if use_cache:
                        use_cache = False
                        current_max_new_tokens = max(96, int(current_max_new_tokens * 0.75))
                        continue

                    current_max_new_tokens = max(96, int(current_max_new_tokens * 0.6))
                    current_max_input_tokens = max(2048, int(current_max_input_tokens * 0.9))
                    current_char_limit = max(6000, int(current_char_limit * 0.85))
                    current_image_max = (640, 360)

            generation_time = time.time() - start_time
            print(f"DEBUG: Generation took {generation_time:.2f}s", flush=True)

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self._processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return output_text[0] if output_text else ""
        finally:
            # Aggressive memory cleanup
            if inputs is not None:
                del inputs
            if generated_ids is not None:
                del generated_ids
            if self._device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    def _extract_json(self, text: str) -> str:
        stack = []
        start_index = text.find('{')
        if start_index == -1:
            return text

        for i, char in enumerate(text[start_index:], start=start_index):
            if char == '{':
                stack.append(char)
            elif char == '}':
                if stack:
                    stack.pop()
                if not stack:
                    return text[start_index: i + 1]
        return text[start_index:]

    def _clean_json_output(self, data: Any) -> Any:
        if isinstance(data, dict):
            if "action" in data and isinstance(data["action"], list):
                new_actions = []
                i = 0
                while i < len(data["action"]):
                    item = data["action"][i]

                    # Handle case where action is split: ["click", {"index": 1}]
                    if isinstance(item, str) and i + 1 < len(data["action"]):
                        next_item = data["action"][i+1]
                        if isinstance(next_item, dict):
                            # Check if next_item looks like params (doesn't have action keys as top level)
                            # Simple heuristic: treat as params for the string action
                            new_actions.append({item: next_item})
                            i += 2
                            continue

                    if isinstance(item, dict):
                        item.pop("type", None)

                        key_mapping = {
                            "click_element": "click",
                            "input_text": "input",
                            "navigate_browser": "navigate",
                            "open_tab": "new_tab",
                            "switch_tab": "switch",
                            "scroll_element": "scroll",
                            "done_task": "done",
                            "search_google": "search",
                            "go_back": "go_back",
                            "send_keys": "send_keys",
                            "find_element": "find_text",
                        }

                        new_action = {}
                        for k, v in item.items():
                            if k in key_mapping:
                                new_action[key_mapping[k]] = v
                            else:
                                new_action[k] = v
                        new_actions.append(new_action)
                        i += 1
                    else:
                        # Keep as is if we can't fix it, or skip strings that were not part of a pair
                        if not isinstance(item, str):
                            new_actions.append(item)
                        i += 1
                data["action"] = new_actions

            for key, value in data.items():
                self._clean_json_output(value)
        elif isinstance(data, list):
            for item in data:
                self._clean_json_output(item)
        return data

    async def ainvoke(
        self,
        messages: list[BaseMessage],
        output_format: type[T] | None = None,
        **kwargs: Any,
    ) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
        self._ensure_loaded()
        serialized = self._serialize_messages(messages, output_format)
        # Increase default tokens to avoid cutoff
        max_new_tokens = int(kwargs.get("max_new_tokens", 256))
        completion_text = await asyncio.to_thread(self._generate, serialized, max_new_tokens)
        if output_format is None:
            return ChatInvokeCompletion(completion=completion_text, usage=None)

        parsed_text = self._extract_json(completion_text)

        # Pre-parse repair: Insert missing comma between "thinking" and "action"
        parsed_text = re.sub(r'("\s*)\n(\s*"action")', r'\1,\n\2', parsed_text)

        try:
            # Try direct validation first
            completion = output_format.model_validate_json(parsed_text)
        except ValidationError as first_err:
            # If validation fails, try to load as dict, clean, and re-validate
            try:
                data = json.loads(parsed_text)
                cleaned_data = self._clean_json_output(data)
                completion = output_format.model_validate(cleaned_data)
            except Exception:
                try:
                    repair_msg = {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Your previous output did not match the required JSON schema.\n"
                                    f"Validation error: {first_err}\n"
                                    "Return ONLY valid JSON that matches the schema exactly."
                                ),
                            }
                        ],
                    }
                    repaired_text = await asyncio.to_thread(
                        self._generate, serialized +
                        [repair_msg], min(max_new_tokens, 192)
                    )
                    repaired_json = self._extract_json(repaired_text)
                    repaired_json = re.sub(
                        r'("\s*)\n(\s*"action")', r'\1,\n\2', repaired_json)
                    completion = output_format.model_validate_json(
                        repaired_json)
                except Exception:
                    print(
                        f"JSON Validation failed. Raw output: {completion_text}")
                    raise

        return ChatInvokeCompletion(completion=completion, usage=None)


async def example():
    browser = Browser(
        # use_cloud=True,  # Uncomment to use a stealth browser on Browser Use Cloud
    )

    llm = Qwen3VLLocal()

    agent = Agent(
        task="Download service now icon from bing, the picture format should be png",
        llm=llm,
        browser=browser,
        llm_timeout=120 * 5,
        step_timeout=120 * 5,
        vision_detail_level="low",
        llm_screenshot_size=(1280, 720),
    )

    try:
        history = await agent.run()
        return history
    finally:
        close = getattr(browser, "close", None)
        if close is not None:
            result = close()

if __name__ == "__main__":
    asyncio.run(example())
