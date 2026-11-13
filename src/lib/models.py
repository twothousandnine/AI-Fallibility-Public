import re
from typing import List

MODEL_DISPLAY_NAMES = {
    "us.anthropic.claude-opus-4-7": "Claude Opus 4.7",
    "us.anthropic.claude-sonnet-4-6": "Claude Sonnet 4.6",
    "us.meta.llama4-maverick-17b-instruct-v1:0": "Llama 4 Maverick",
    "us.meta.llama4-scout-17b-instruct-v1:0": "Llama 4 Scout",
    "us.deepseek.r1-v1:0": "DeepSeek R1",
    "deepseek.v3.2": "DeepSeek V3.2",
    "us.amazon.nova-pro-v1:0": "Nova Pro V1",
    "us.meta.llama3-3-70b-instruct-v1:0": "Llama 3.3 70B",
    "qwen.qwen3-next-80b-a3b": "Qwen3 Next 80B A3B",
    "mistral.mistral-large-3-675b-instruct": "Mistral Large 3",
    "openai.gpt-5.6-luna": "GPT 5.6",
    "openai.gpt-5.4": "GPT 5.4 (thinking)",
    "google.gemma-4-31b": "Gemma 4 31B",
    "xai.grok-4.3": "Grok 4.3",
    "z-ai/glm-5.3": "GLM 5.3",
    "deepseek/deepseek-v4-pro-0813": "DeepSeek V4 Pro",
    "moonshotai/kimi-k3": "Kimi K3",
    "qwen/qwen3.8-max-0902": "Qwen3.8 Max",
    "meta/muse-glimmer-30b": "Muse Glimmer 30B",
    "google/gemini-3.8-flash": "Gemini 3.8 Flash",
}

SHORT_MODEL_DISPLAY_NAMES = {
    "Claude Opus 4.7": "Opus 4.7",
    "Claude Sonnet 4.6": "Sonnet 4.6",
    "Llama 3.3 70B": "Llama 3.3",
    "Qwen3 Next 80B A3B": "Qwen3 Next",
    "Gemma 4 31B": "Gemma 4",
    "GPT 5.4 (thinking)": "GPT 5.4",
    "Muse Glimmer 30B": "Muse Glimmer",
}

OPEN_WEIGHTS_BY_MODEL_ID = {
    "deepseek.v3.2": True,
    "google.gemma-4-31b": True,
    "mistral.mistral-large-3-675b-instruct": True,
    "openai.gpt-5.4": False,
    "openai.gpt-5.6-luna": False,
    "qwen.qwen3-next-80b-a3b": True,
    "us.amazon.nova-pro-v1:0": False,
    "us.anthropic.claude-opus-4-7": False,
    "us.anthropic.claude-sonnet-4-6": False,
    "us.deepseek.r1-v1:0": True,
    "us.meta.llama3-3-70b-instruct-v1:0": True,
    "us.meta.llama4-maverick-17b-instruct-v1:0": True,
    "us.meta.llama4-scout-17b-instruct-v1:0": True,
    "xai.grok-4.3": False,
    "z-ai/glm-5.3": True,
    "deepseek/deepseek-v4-pro-0813": True,
    "moonshotai/kimi-k3": True,
    "qwen/qwen3.8-max-0902": True,
    "meta/muse-glimmer-30b": True,
    "google/gemini-3.8-flash": False,
}

MODEL_IDS: List[str] = list(MODEL_DISPLAY_NAMES.keys())


def normalize_model_identifier(model_id: str) -> str:
    return re.sub(r"[^a-z0-9]", "", model_id.lower())


def display_name(model_id: str) -> str:
    return MODEL_DISPLAY_NAMES[model_id]


def short_display_name(model_id: str) -> str:
    full_name = display_name(model_id)
    return SHORT_MODEL_DISPLAY_NAMES.get(full_name, full_name)


def is_open_weights_model(model_id: str) -> bool:
    return OPEN_WEIGHTS_BY_MODEL_ID[model_id]
