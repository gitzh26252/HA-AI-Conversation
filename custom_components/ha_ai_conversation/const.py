"""Constants for the HA AI conversation integration."""

from __future__ import annotations

import logging

DOMAIN = "ha_ai_conversation"
LOGGER: logging.Logger = logging.getLogger(__package__)

CONF_API_BASE = "api_base"
CONF_API_MODE = "api_mode"
CONF_CHAT_MODEL = "chat_model"
CONF_MAX_TOKENS = "max_tokens"
CONF_PROMPT = "prompt"
CONF_RECOMMENDED = "recommended"
CONF_TEMPERATURE = "temperature"
CONF_TOP_P = "top_p"

API_MODE_AUTO = "auto"
API_MODE_OPENAI_RESPONSES = "openai_responses"
API_MODE_OPENAI_CHAT = "openai_chat_completions"
API_MODE_ANTHROPIC = "anthropic_messages"
API_MODE_GEMINI = "gemini_generate_content"

RECOMMENDED_CHAT_MODEL = "gpt-4o-mini"
RECOMMENDED_MAX_TOKENS = 512
RECOMMENDED_TEMPERATURE = 0.7
RECOMMENDED_TOP_P = 1.0

RECOMMENDED_OPTIONS = {
    CONF_RECOMMENDED: True,
    CONF_PROMPT: (
        "You are a Home Assistant voice and text assistant. "
        "Answer briefly, act safely, and use exposed tools when needed."
    ),
}