"""Unified client adapters for multiple LLM provider APIs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import httpx

from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant
from homeassistant.helpers.httpx_client import get_async_client

from .const import (
    API_MODE_ANTHROPIC,
    API_MODE_AUTO,
    API_MODE_GEMINI,
    API_MODE_OPENAI_CHAT,
    API_MODE_OPENAI_RESPONSES,
    CONF_API_BASE,
    CONF_API_MODE,
    CONF_CHAT_MODEL,
)


class APIError(Exception):
    """Base API error."""


class ValidationError(ValueError):
    """Validation error with a user-facing message."""

    def __init__(self, error_key: str, detail: str | None = None) -> None:
        """Initialize validation error."""
        super().__init__(error_key)
        self.error_key = error_key
        self.detail = detail


@dataclass(slots=True)
class ChatMessage:
    """Internal normalized chat message."""

    role: str
    content: str
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    tool_result: dict[str, Any] | str | None = None


@dataclass(slots=True)
class UnifiedClient:
    """Unified chat client."""

    http_client: httpx.AsyncClient
    base_url: str
    api_key: str
    model: str
    mode: str
    provider_name: str

    async def validate(self) -> None:
        """Validate credentials and endpoint with a cheap provider-specific call."""
        if self.mode == API_MODE_OPENAI_RESPONSES:
            await self._post_openai_responses("ping", max_tokens=1)
            return
        if self.mode == API_MODE_OPENAI_CHAT:
            await self._post_openai_chat([{"role": "user", "content": "ping"}], max_tokens=1)
            return
        if self.mode == API_MODE_ANTHROPIC:
            await self._post_anthropic([{"role": "user", "content": "ping"}], max_tokens=1)
            return
        if self.mode == API_MODE_GEMINI:
            await self._post_gemini([{"role": "user", "parts": [{"text": "ping"}]}], max_tokens=1)
            return
        raise ValueError("unsupported_api_mode")

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        tools: list[dict[str, Any]] | None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Send a completion request and return text plus tool calls."""
        if self.mode == API_MODE_OPENAI_RESPONSES:
            return await self._complete_openai_responses(messages, max_tokens, temperature, top_p, tools)
        if self.mode == API_MODE_OPENAI_CHAT:
            return await self._complete_openai_chat(messages, max_tokens, temperature, top_p, tools)
        if self.mode == API_MODE_ANTHROPIC:
            return await self._complete_anthropic(messages, max_tokens, temperature, top_p, tools)
        if self.mode == API_MODE_GEMINI:
            return await self._complete_gemini(messages, max_tokens, temperature, top_p, tools)
        raise ValueError("unsupported_api_mode")

    async def _request(self, method: str, url: str, *, headers: dict[str, str], json_body: dict[str, Any]) -> dict[str, Any]:
        response = await self.http_client.request(method, url, headers=headers, json=json_body, timeout=20.0)
        if response.status_code in (401, 403):
            raise ValueError("invalid_auth")
        if response.status_code >= 400:
            try:
                payload = response.json()
            except Exception:
                payload = {"error": {"message": response.text}}
            message = payload.get("error", {}).get("message") or response.text
            raise APIError(message)
        return response.json()

    async def _post_openai_responses(self, prompt: str, *, max_tokens: int) -> dict[str, Any]:
        return await self._request(
            "POST",
            self.base_url.rstrip("/") + "/responses",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json_body={"model": self.model, "input": prompt, "max_output_tokens": max_tokens},
        )

    async def _post_openai_chat(self, messages: list[dict[str, Any]], *, max_tokens: int) -> dict[str, Any]:
        return await self._request(
            "POST",
            self.base_url.rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json_body={"model": self.model, "messages": messages, "max_tokens": max_tokens},
        )

    async def _post_anthropic(self, messages: list[dict[str, Any]], *, max_tokens: int) -> dict[str, Any]:
        return await self._request(
            "POST",
            self.base_url.rstrip("/") + "/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json_body={"model": self.model, "messages": messages, "max_tokens": max_tokens},
        )

    async def _post_gemini(self, contents: list[dict[str, Any]], *, max_tokens: int) -> dict[str, Any]:
        return await self._request(
            "POST",
            self.base_url.rstrip("/") + f"/models/{self.model}:generateContent?key={self.api_key}",
            headers={"Content-Type": "application/json"},
            json_body={"contents": contents, "generationConfig": {"maxOutputTokens": max_tokens}},
        )

    async def _complete_openai_responses(self, messages: list[ChatMessage], max_tokens: int, temperature: float, top_p: float, tools: list[dict[str, Any]] | None) -> tuple[str, list[dict[str, Any]]]:
        input_items: list[dict[str, Any]] = []
        for message in messages:
            if message.tool_result is not None and message.tool_call_id:
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": message.tool_call_id,
                        "output": json.dumps(message.tool_result),
                    }
                )
                continue
            input_items.append({"role": message.role, "content": message.content})
            if message.tool_calls:
                input_items.extend(message.tool_calls)

        payload: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            "max_output_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "store": False,
        }
        if tools:
            payload["tools"] = tools

        data = await self._request(
            "POST",
            self.base_url.rstrip("/") + "/responses",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json_body=payload,
        )
        output_text = data.get("output_text", "")
        tool_calls: list[dict[str, Any]] = []
        for item in data.get("output", []):
            if item.get("type") == "function_call":
                tool_calls.append(
                    {
                        "id": item["call_id"],
                        "type": "function",
                        "function": {
                            "name": item["name"],
                            "arguments": item.get("arguments", "{}"),
                        },
                    }
                )
        return output_text, tool_calls

    async def _complete_openai_chat(self, messages: list[ChatMessage], max_tokens: int, temperature: float, top_p: float, tools: list[dict[str, Any]] | None) -> tuple[str, list[dict[str, Any]]]:
        payload_messages: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "tool" and message.tool_call_id:
                payload_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.tool_call_id,
                        "content": json.dumps(message.tool_result),
                    }
                )
                continue
            payload: dict[str, Any] = {"role": message.role, "content": message.content}
            if message.tool_calls:
                payload["tool_calls"] = message.tool_calls
            payload_messages.append(payload)

        body: dict[str, Any] = {
            "model": self.model,
            "messages": payload_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        data = await self._request(
            "POST",
            self.base_url.rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json_body=body,
        )
        message = data["choices"][0]["message"]
        return message.get("content", "") or "", message.get("tool_calls", []) or []

    async def _complete_anthropic(self, messages: list[ChatMessage], max_tokens: int, temperature: float, top_p: float, tools: list[dict[str, Any]] | None) -> tuple[str, list[dict[str, Any]]]:
        system_blocks: list[str] = []
        anthropic_messages: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "system":
                system_blocks.append(message.content)
                continue
            if message.role == "tool" and message.tool_call_id:
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": message.tool_call_id,
                                "content": json.dumps(message.tool_result),
                            }
                        ],
                    }
                )
                continue
            content: list[dict[str, Any] | str] = [message.content]
            if message.tool_calls:
                content = []
                if message.content:
                    content.append({"type": "text", "text": message.content})
                for tool_call in message.tool_calls:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tool_call["id"],
                            "name": tool_call["function"]["name"],
                            "input": json.loads(tool_call["function"]["arguments"]),
                        }
                    )
            anthropic_messages.append({"role": message.role, "content": content})

        body: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if system_blocks:
            body["system"] = "\n\n".join(system_blocks)
        if tools:
            body["tools"] = [
                {
                    "name": tool["function"]["name"],
                    "description": tool["function"].get("description", ""),
                    "input_schema": tool["function"]["parameters"],
                }
                for tool in tools
            ]

        data = await self._request(
            "POST",
            self.base_url.rstrip("/") + "/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json_body=body,
        )
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for item in data.get("content", []):
            if item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            elif item.get("type") == "tool_use":
                tool_calls.append(
                    {
                        "id": item["id"],
                        "type": "function",
                        "function": {
                            "name": item["name"],
                            "arguments": json.dumps(item.get("input", {})),
                        },
                    }
                )
        return "".join(text_parts), tool_calls

    async def _complete_gemini(self, messages: list[ChatMessage], max_tokens: int, temperature: float, top_p: float, tools: list[dict[str, Any]] | None) -> tuple[str, list[dict[str, Any]]]:
        contents: list[dict[str, Any]] = []
        system_prompt: str | None = None
        for message in messages:
            if message.role == "system":
                system_prompt = f"{system_prompt}\n{message.content}" if system_prompt else message.content
                continue
            parts: list[dict[str, Any]] = []
            if message.role == "tool" and message.tool_call_id:
                parts.append(
                    {
                        "functionResponse": {
                            "name": message.tool_call_id,
                            "response": {"content": message.tool_result},
                        }
                    }
                )
                contents.append({"role": "user", "parts": parts})
                continue
            if message.content:
                parts.append({"text": message.content})
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    parts.append(
                        {
                            "functionCall": {
                                "name": tool_call["function"]["name"],
                                "args": json.loads(tool_call["function"]["arguments"]),
                            }
                        }
                    )
            role = "model" if message.role == "assistant" else "user"
            contents.append({"role": role, "parts": parts})

        body: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "topP": top_p,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_prompt:
            body["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        if tools:
            body["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": tool["function"]["name"],
                            "description": tool["function"].get("description", ""),
                            "parameters": tool["function"]["parameters"],
                        }
                        for tool in tools
                    ]
                }
            ]

        data = await self._request(
            "POST",
            self.base_url.rstrip("/") + f"/models/{self.model}:generateContent?key={self.api_key}",
            headers={"Content-Type": "application/json"},
            json_body=body,
        )
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for part in parts:
            if "text" in part:
                text_parts.append(part["text"])
            elif "functionCall" in part:
                function_call = part["functionCall"]
                tool_calls.append(
                    {
                        "id": function_call["name"],
                        "type": "function",
                        "function": {
                            "name": function_call["name"],
                            "arguments": json.dumps(function_call.get("args", {})),
                        },
                    }
                )
        return "".join(text_parts), tool_calls


def _guess_modes(base_url: str) -> list[str]:
    normalized = base_url.lower().rstrip("/")
    if "anthropic" in normalized:
        return [API_MODE_ANTHROPIC]
    if "generativelanguage.googleapis.com" in normalized or "gemini" in normalized:
        return [API_MODE_GEMINI]
    if normalized.endswith("/responses"):
        return [API_MODE_OPENAI_RESPONSES]
    if normalized.endswith("/chat/completions"):
        return [API_MODE_OPENAI_CHAT]
    if normalized.endswith("/messages"):
        return [API_MODE_ANTHROPIC]
    return [API_MODE_OPENAI_RESPONSES, API_MODE_OPENAI_CHAT, API_MODE_ANTHROPIC, API_MODE_GEMINI]


def _normalize_base_url(base_url: str, mode: str) -> str:
    normalized = base_url.rstrip("/")
    if mode == API_MODE_OPENAI_RESPONSES and normalized.endswith("/responses"):
        return normalized[: -len("/responses")]
    if mode == API_MODE_OPENAI_CHAT and normalized.endswith("/chat/completions"):
        return normalized[: -len("/chat/completions")]
    if mode == API_MODE_ANTHROPIC and normalized.endswith("/messages"):
        return normalized[: -len("/messages")]
    return normalized


async def detect_and_validate_client(hass: HomeAssistant, data: dict[str, Any]) -> UnifiedClient:
    """Detect API mode when needed and validate connectivity."""
    requested_mode = data.get(CONF_API_MODE, API_MODE_AUTO)
    modes = [requested_mode] if requested_mode != API_MODE_AUTO else _guess_modes(data[CONF_API_BASE])
    http_client = get_async_client(hass)

    last_error: Exception | None = None
    for mode in modes:
        client = UnifiedClient(
            http_client=http_client,
            base_url=_normalize_base_url(data[CONF_API_BASE], mode),
            api_key=data[CONF_API_KEY],
            model=data[CONF_CHAT_MODEL],
            mode=mode,
            provider_name={
                API_MODE_OPENAI_RESPONSES: "OpenAI-compatible",
                API_MODE_OPENAI_CHAT: "OpenAI-compatible",
                API_MODE_ANTHROPIC: "Anthropic-compatible",
                API_MODE_GEMINI: "Gemini-compatible",
            }[mode],
        )
        try:
            await client.validate()
        except Exception as err:
            last_error = err
            continue
        return client

    if isinstance(last_error, ValueError):
        raise last_error

    detail = None
    if last_error is not None:
        detail = str(last_error).strip() or type(last_error).__name__
    raise ValidationError("cannot_connect", detail) from last_error