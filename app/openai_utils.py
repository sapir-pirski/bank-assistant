from __future__ import annotations

from typing import Any

from openai import BadRequestError, OpenAI

from app.config import Settings


def build_openai_client(settings: Settings) -> OpenAI:
    settings.require_openai_key()
    return OpenAI(api_key=settings.openai_api_key)


def create_response_with_temperature(
    client: OpenAI,
    *,
    model: str,
    instructions: str,
    input: str,
    max_output_tokens: int,
    store: bool,
    temperature: float | None,
    extra_request: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    request: dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": input,
        "max_output_tokens": max_output_tokens,
        "store": store,
    }
    metadata: dict[str, Any] = {
        "temperature_requested": temperature,
        "temperature_sent": None,
        "temperature_fallback": False,
        "model": model,
    }

    if extra_request:
        request.update(extra_request)

    if temperature is not None:
        request["temperature"] = temperature
        metadata["temperature_sent"] = temperature

    try:
        response = client.responses.create(**request)
        metadata["usage"] = response_usage(response)
        return response, metadata
    except BadRequestError as exc:
        message = str(exc)
        if temperature is not None and "temperature" in message and "Unsupported parameter" in message:
            request.pop("temperature", None)
            metadata["temperature_sent"] = None
            metadata["temperature_fallback"] = True
            metadata["temperature_error"] = message
            response = client.responses.create(**request)
            metadata["usage"] = response_usage(response)
            return response, metadata
        raise


def extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text).strip()

    parts: list[str] = []
    for item in _get(response, "output", []) or []:
        for content in _get(item, "content", []) or []:
            if _get(content, "type", None) == "output_text":
                text = _get(content, "text", "")
                if text:
                    parts.append(str(text))
    return "\n".join(parts).strip()


def response_usage(response: Any) -> dict[str, int | None]:
    usage = _get(response, "usage", None)
    if not usage:
        return {}
    return {
        "input_tokens": _get(usage, "input_tokens", None),
        "output_tokens": _get(usage, "output_tokens", None),
        "total_tokens": _get(usage, "total_tokens", None),
    }


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)
