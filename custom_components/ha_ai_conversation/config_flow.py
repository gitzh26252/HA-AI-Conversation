"""Config flow for the HA AI conversation integration."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

import voluptuous as vol

from homeassistant.config_entries import ConfigEntry, ConfigFlow, ConfigFlowResult, OptionsFlow
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API
from homeassistant.helpers import llm
from homeassistant.helpers.selector import NumberSelector, NumberSelectorConfig, SelectOptionDict, SelectSelector, SelectSelectorConfig, SelectSelectorMode, TemplateSelector
from homeassistant.helpers.typing import VolDictType

from .api import detect_and_validate_client
from .const import API_MODE_ANTHROPIC, API_MODE_AUTO, API_MODE_GEMINI, API_MODE_OPENAI_CHAT, API_MODE_OPENAI_RESPONSES, CONF_API_BASE, CONF_API_MODE, CONF_CHAT_MODEL, CONF_MAX_TOKENS, CONF_PROMPT, CONF_RECOMMENDED, CONF_TEMPERATURE, CONF_TOP_P, DOMAIN, RECOMMENDED_CHAT_MODEL, RECOMMENDED_MAX_TOKENS, RECOMMENDED_OPTIONS, RECOMMENDED_TEMPERATURE, RECOMMENDED_TOP_P

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_API_KEY): str,
        vol.Required(CONF_API_BASE): str,
        vol.Required(CONF_CHAT_MODEL, default=RECOMMENDED_CHAT_MODEL): str,
        vol.Required(CONF_API_MODE, default=API_MODE_AUTO): SelectSelector(
            SelectSelectorConfig(
                options=[
                    SelectOptionDict(value=API_MODE_AUTO, label="Auto detect"),
                    SelectOptionDict(value=API_MODE_OPENAI_RESPONSES, label="OpenAI Responses"),
                    SelectOptionDict(value=API_MODE_OPENAI_CHAT, label="OpenAI Chat Completions"),
                    SelectOptionDict(value=API_MODE_ANTHROPIC, label="Anthropic Messages"),
                    SelectOptionDict(value=API_MODE_GEMINI, label="Gemini Generate Content"),
                ],
                mode=SelectSelectorMode.DROPDOWN,
            )
        ),
    }
)


async def validate_input(hass, data: dict[str, Any]) -> dict[str, Any]:
    """Validate user input and detect provider mode when needed."""
    client = await detect_and_validate_client(hass, data)
    return {
        CONF_API_MODE: client.mode,
        "title": f"{client.provider_name} ({data[CONF_CHAT_MODEL]})",
    }


class HAAIConversationConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for the HA AI conversation integration."""

    VERSION = 2

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            try:
                info = await validate_input(self.hass, user_input)
            except ValueError as err:
                errors["base"] = str(err)
            except Exception:
                errors["base"] = "unknown"
            else:
                user_input[CONF_API_MODE] = info[CONF_API_MODE]
                await self.async_set_unique_id(f"{user_input[CONF_API_BASE].rstrip('/')}-{user_input[CONF_CHAT_MODEL]}")
                self._abort_if_unique_id_configured()
                return self.async_create_entry(
                    title=info["title"],
                    data=user_input,
                    options={**RECOMMENDED_OPTIONS, CONF_LLM_HASS_API: llm.LLM_API_ASSIST},
                )

        return self.async_show_form(
            step_id="user",
            data_schema=STEP_USER_DATA_SCHEMA,
            errors=errors,
            description_placeholders={"example_url": "https://api.openai.com/v1 or https://your-endpoint/openai/v1"},
        )

    @staticmethod
    def async_get_options_flow(config_entry: ConfigEntry) -> OptionsFlow:
        """Create the options flow."""
        return HAAIConversationOptionsFlow(config_entry)


class HAAIConversationOptionsFlow(OptionsFlow):
    """Options flow for the HA AI conversation integration."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize options flow."""
        self.last_rendered_recommended = config_entry.options.get(CONF_RECOMMENDED, False)

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Manage the options."""
        options: dict[str, Any] | MappingProxyType[str, Any] = self.config_entry.options

        if user_input is not None:
            if user_input[CONF_RECOMMENDED] == self.last_rendered_recommended:
                if not user_input.get(CONF_LLM_HASS_API):
                    user_input.pop(CONF_LLM_HASS_API, None)
                return self.async_create_entry(title="", data=user_input)

            self.last_rendered_recommended = user_input[CONF_RECOMMENDED]
            options = {
                CONF_RECOMMENDED: user_input[CONF_RECOMMENDED],
                CONF_PROMPT: user_input.get(CONF_PROMPT, RECOMMENDED_OPTIONS[CONF_PROMPT]),
                CONF_LLM_HASS_API: user_input.get(CONF_LLM_HASS_API),
            }

        schema = ai_config_option_schema(options)
        return self.async_show_form(step_id="init", data_schema=vol.Schema(schema))


def ai_config_option_schema(options: dict[str, Any] | MappingProxyType[str, Any]) -> VolDictType:
    """Return options schema."""
    recommended = options.get(CONF_RECOMMENDED, True)
    schema: VolDictType = {
        vol.Required(CONF_RECOMMENDED, default=recommended): bool,
        vol.Optional(CONF_LLM_HASS_API, default=options.get(CONF_LLM_HASS_API)): llm.llm_api_selector,
        vol.Required(CONF_PROMPT, default=options.get(CONF_PROMPT, RECOMMENDED_OPTIONS[CONF_PROMPT])): TemplateSelector(),
    }

    if not recommended:
        schema.update(
            {
                vol.Required(CONF_CHAT_MODEL, default=options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)): str,
                vol.Required(CONF_MAX_TOKENS, default=options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS)): NumberSelector(NumberSelectorConfig(min=1, max=65536, mode="box")),
                vol.Required(CONF_TEMPERATURE, default=options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE)): NumberSelector(NumberSelectorConfig(min=0, max=2, step=0.1, mode="slider")),
                vol.Required(CONF_TOP_P, default=options.get(CONF_TOP_P, RECOMMENDED_TOP_P)): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05, mode="slider")),
            }
        )

    return schema