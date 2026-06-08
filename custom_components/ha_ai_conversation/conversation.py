"""Conversation support for universal AI providers."""

from __future__ import annotations

import json
from typing import Any, Literal

from voluptuous_openapi import convert

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, intent, llm
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import HAAIConversationConfigEntry
from .api import ChatMessage
from .const import CONF_MAX_TOKENS, CONF_PROMPT, CONF_TEMPERATURE, CONF_TOP_P, DOMAIN, LOGGER, RECOMMENDED_MAX_TOKENS, RECOMMENDED_TEMPERATURE, RECOMMENDED_TOP_P

MAX_TOOL_ITERATIONS = 10


async def async_setup_entry(hass: HomeAssistant, config_entry: HAAIConversationConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    """Set up conversation entities."""
    async_add_entities([UniversalConversationEntity(config_entry)])


def _format_tool(tool: llm.Tool, custom_serializer) -> dict[str, Any]:
    """Format HA tool into OpenAI-compatible shape."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": convert(tool.parameters, custom_serializer=custom_serializer),
        },
    }


def _message_from_content(content: conversation.Content) -> list[ChatMessage]:
    messages: list[ChatMessage] = []
    if isinstance(content, conversation.ToolResultContent):
        return [ChatMessage(role="tool", content="", tool_call_id=content.tool_call_id, tool_result=content.tool_result)]

    if content.content:
        role: Literal["user", "assistant", "system"] = content.role
        messages.append(ChatMessage(role=role, content=content.content))

    if isinstance(content, conversation.AssistantContent) and content.tool_calls:
        tool_calls = [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.tool_name,
                    "arguments": json.dumps(tool_call.tool_args),
                },
            }
            for tool_call in content.tool_calls
        ]
        if messages:
            messages[-1].tool_calls = tool_calls
        else:
            messages.append(ChatMessage(role="assistant", content="", tool_calls=tool_calls))
    return messages


class UniversalConversationEntity(conversation.ConversationEntity, conversation.AbstractConversationAgent):
    """Universal conversation agent."""

    _attr_has_entity_name = True
    _attr_name = None
    _attr_supports_streaming = False

    def __init__(self, entry: HAAIConversationConfigEntry) -> None:
        self.entry = entry
        self._attr_unique_id = entry.entry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="Universal AI",
            model="Conversation Agent",
            entry_type=dr.DeviceEntryType.SERVICE,
        )
        if self.entry.options.get(CONF_LLM_HASS_API):
            self._attr_supported_features = conversation.ConversationEntityFeature.CONTROL

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        return MATCH_ALL

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)
        self.entry.async_on_unload(self.entry.add_update_listener(self._async_entry_update_listener))

    async def async_will_remove_from_hass(self) -> None:
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    async def _async_handle_message(self, user_input: conversation.ConversationInput, chat_log: conversation.ChatLog) -> conversation.ConversationResult:
        options = self.entry.options

        try:
            await chat_log.async_update_llm_data(DOMAIN, user_input, options.get(CONF_LLM_HASS_API), options.get(CONF_PROMPT))
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        await self._async_handle_chat_log(chat_log)

        intent_response = intent.IntentResponse(language=user_input.language)
        assert type(chat_log.content[-1]) is conversation.AssistantContent
        intent_response.async_set_speech(chat_log.content[-1].content or "")
        return conversation.ConversationResult(response=intent_response, conversation_id=chat_log.conversation_id, continue_conversation=chat_log.continue_conversation)

    async def _async_handle_chat_log(self, chat_log: conversation.ChatLog) -> None:
        options = self.entry.options
        client = self.entry.runtime_data

        tools: list[dict[str, Any]] | None = None
        if chat_log.llm_api:
            tools = [_format_tool(tool, chat_log.llm_api.custom_serializer) for tool in chat_log.llm_api.tools]

        messages = [message for content in chat_log.content for message in _message_from_content(content)]

        for _ in range(MAX_TOOL_ITERATIONS):
            try:
                text, tool_calls = await client.complete(
                    messages,
                    max_tokens=options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
                    temperature=options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
                    top_p=options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
                    tools=tools,
                )
            except Exception as err:
                LOGGER.error("Error talking to AI provider: %s", err)
                raise HomeAssistantError("Error talking to AI provider") from err

            assistant = conversation.AssistantContent(agent_id=self.entity_id, content=text or None)
            if tool_calls:
                assistant.tool_calls = [
                    llm.ToolInput(
                        id=tool_call["id"],
                        tool_name=tool_call["function"]["name"],
                        tool_args=json.loads(tool_call["function"]["arguments"]),
                    )
                    for tool_call in tool_calls
                ]
            chat_log.async_add_content(assistant)

            if not tool_calls:
                break

            await chat_log.async_provide_llm_data(assistant, self.entity_id)
            if not chat_log.unresponded_tool_results:
                break
            messages = [message for content in chat_log.content for message in _message_from_content(content)]

    async def _async_entry_update_listener(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        await hass.config_entries.async_reload(entry.entry_id)