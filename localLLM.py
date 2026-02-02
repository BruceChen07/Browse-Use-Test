from transformers import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLProcessor,
    Qwen2VLImageProcessorFast,
    Qwen3VLVideoProcessor,
    AutoTokenizer,
    AutoConfig
)
import torch

# Define model ID
model_id = "Qwen/Qwen3-VL-2B-Instruct"

print(f"Loading configuration for {model_id}...")
# 1. Load and Patch Configuration
try:
    config = AutoConfig.from_pretrained(model_id)

    # FIX: Patch missing pad_token_id in text_config which causes AttributeError
    if hasattr(config, "text_config"):
        # Check if pad_token_id is missing or None
        if not hasattr(config.text_config, "pad_token_id") or config.text_config.pad_token_id is None:
            print("Patching missing pad_token_id in text_config...")
            # Try to get from main config
            pad_token_id = getattr(config, "pad_token_id", None)
            # If not in main, try eos_token_id
            if pad_token_id is None:
                pad_token_id = getattr(config, "eos_token_id", None)
            # If still None, fallback to a reasonable default (e.g. Qwen2 default)
            if pad_token_id is None:
                pad_token_id = 151643

            # Manually set the attribute
            config.text_config.pad_token_id = pad_token_id
            print(f"Set text_config.pad_token_id to {pad_token_id}")

except Exception as e:
    print(f"Warning: Failed to load/patch config: {e}")
    config = None

# 2. Load Model
print("Loading model...")
try:
    # Use the patched config if available
    load_kwargs = {"dtype": "auto", "device_map": "auto"}
    if config:
        load_kwargs["config"] = config

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        **load_kwargs
    )
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# 3. Load Processor
print("Loading processor...")
try:
    # Try direct loading first
    processor = Qwen3VLProcessor.from_pretrained(model_id)
except Exception:
    print("Direct processor loading failed, constructing manually...")
    # Manual fallback construction
    try:
        image_processor = Qwen2VLImageProcessorFast.from_pretrained(model_id)
        video_processor = Qwen3VLVideoProcessor.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        processor = Qwen3VLProcessor(
            image_processor=image_processor,
            video_processor=video_processor,
            tokenizer=tokenizer
        )
    except Exception as e:
        print(f"Manual processor construction failed: {e}")
        raise

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
print("Preparing inputs...")
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
print("Generating...")
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print("Output:")
print(output_text)
