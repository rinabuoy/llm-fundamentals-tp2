"""One `chat()` function for notebooks 02–04, backed by Ollama, Google Gemini or any OpenAI-compatible API.

    import llm_client as llm
    llm.configure("ollama")   # local model through Ollama
    llm.configure("gemini")   # Google Gemini, using GEMINI_KEY from the environment or a .env file
    llm.configure("api")      # any OpenAI-compatible cloud API (OpenAI, Groq, OpenRouter, ...)

    response = llm.chat(messages=[...], tools=[...])
    response.message.content
    response.message.tool_calls[0].function.name / .arguments (a dict)

All backends return the same shape as `ollama.chat`, so the notebooks' code does not change when you switch.
Settings are read from environment variables (or a `.env` file in this folder or a parent) unless you pass
them to `configure()`:

    GEMINI_KEY        your Google AI Studio key, for the "gemini" backend
    GEMINI_MODEL      optional; defaults to gemini-flash-lite-latest

The "api" backend reads:

    LLM_API_KEY       your provider's API key (falls back to OPENAI_API_KEY; asked for interactively if missing)
    LLM_API_BASE_URL  e.g. https://api.openai.com/v1 (default)
                           https://generativelanguage.googleapis.com/v1beta/openai/   (Google Gemini)
                           https://api.groq.com/openai/v1                             (Groq)
                           https://openrouter.ai/api/v1                               (OpenRouter)
                           http://localhost:11434/v1                                  (Ollama's own OpenAI-compatible endpoint)
    LLM_API_MODEL     the model name, as listed in your provider's docs (it must support tool calling)
"""
import json
import os
from dataclasses import dataclass, field
from getpass import getpass
from typing import Any, Dict, List, Optional

try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))  # picks up GEMINI_KEY etc. from a .env file, if there is one
except ImportError:
    pass

DEFAULT_OLLAMA_MODEL = "gemma4:e2b-mlx"
DEFAULT_API_BASE_URL = "https://api.openai.com/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_GEMINI_MODEL = "gemini-flash-lite-latest"

_config: Dict[str, Any] = {"backend": "ollama", "model": DEFAULT_OLLAMA_MODEL}
_openai_client = None


# ── Response objects shaped like ollama.chat's, so notebook code works with either backend ──
@dataclass
class Function:
    name: str
    arguments: Dict[str, Any]


@dataclass
class ToolCall:
    function: Function
    id: str = ""
    raw: Optional[Dict[str, Any]] = field(default=None, repr=False)  # provider data to send back (e.g. Gemini thought signatures)


@dataclass
class Message:
    role: str = "assistant"
    content: str = ""
    tool_calls: Optional[List[ToolCall]] = None


@dataclass
class ChatResponse:
    message: Message
    model: str = ""
    prompt_eval_count: Optional[int] = None  # prompt tokens, same field name as Ollama
    eval_count: Optional[int] = None         # generated tokens
    raw: Any = field(default=None, repr=False)


def configure(backend: str = "ollama", model: Optional[str] = None,
              base_url: Optional[str] = None, api_key: Optional[str] = None) -> None:
    """Choose where chat() sends requests: "ollama" (local), "gemini" or "api" (OpenAI-compatible cloud API)."""
    global _openai_client
    if backend == "ollama":
        _config.update(backend="ollama", model=model or os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL))
        _openai_client = None
    elif backend == "gemini":
        # Gemini speaks the OpenAI protocol too, so it is the "api" backend with Google's URL and key
        _connect(
            base_url=base_url or GEMINI_BASE_URL,
            api_key=api_key or os.getenv("GEMINI_KEY") or os.getenv("GEMINI_API_KEY") or getpass("Gemini API key: "),
            model=model or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
            backend="gemini",
        )
    elif backend == "api":
        model = model or os.getenv("LLM_API_MODEL")
        if not model:
            raise ValueError(
                "No API model set. Pass configure('api', model='<model name>') or set the LLM_API_MODEL "
                "environment variable to a tool-calling model from your provider's docs."
            )
        _connect(
            base_url=base_url or os.getenv("LLM_API_BASE_URL", DEFAULT_API_BASE_URL),
            api_key=api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or getpass("API key: "),
            model=model,
            backend="api",
        )
    else:
        raise ValueError(f"Unknown backend {backend!r}: use 'ollama', 'gemini' or 'api'")


def _connect(base_url: str, api_key: str, model: str, backend: str) -> None:
    global _openai_client
    from openai import OpenAI

    # Generous retries with backoff: cloud APIs return 429/503 under load (e.g. a whole class at once)
    _openai_client = OpenAI(base_url=base_url, api_key=api_key, max_retries=8)
    _config.update(backend=backend, model=model, base_url=base_url)


def current_model() -> str:
    return _config["model"]


def describe() -> str:
    if _config["backend"] == "ollama":
        return f"Backend: Ollama (local)  |  model: {_config['model']}"
    if _config["backend"] == "gemini":
        return f"Backend: Google Gemini  |  model: {_config['model']}"
    return f"Backend: OpenAI-compatible API at {_config['base_url']}  |  model: {_config['model']}"


def chat(messages: List[Any], tools: Optional[List[Dict]] = None, model: Optional[str] = None,
         options: Optional[Dict[str, Any]] = None) -> Any:
    """Send a chat request to the configured backend. Same arguments and response shape as ollama.chat."""
    model = model or _config["model"]
    if _config["backend"] == "ollama":
        import ollama

        return ollama.chat(model=model, messages=messages, tools=tools, options=options)
    return _chat_openai(messages, tools, model, options or {})


def _chat_openai(messages: List[Any], tools: Optional[List[Dict]], model: str, options: Dict[str, Any]) -> ChatResponse:
    kwargs: Dict[str, Any] = {"model": model, "messages": _to_openai_messages(messages)}
    if tools:
        kwargs["tools"] = tools
    # Map Ollama-style options onto the OpenAI parameters that exist (there is no top_k)
    for ollama_name, openai_name in [("temperature", "temperature"), ("top_p", "top_p"), ("num_predict", "max_tokens"), ("seed", "seed")]:
        if ollama_name in options:
            kwargs[openai_name] = options[ollama_name]

    completion = _openai_client.chat.completions.create(**kwargs)
    choice = completion.choices[0].message
    tool_calls = [
        ToolCall(
            id=tc.id,
            function=Function(name=tc.function.name, arguments=json.loads(tc.function.arguments or "{}")),
            raw=tc.model_dump(exclude_none=True),
        )
        for tc in (choice.tool_calls or [])
    ]
    usage = completion.usage
    return ChatResponse(
        message=Message(content=choice.content or "", tool_calls=tool_calls or None),
        model=completion.model,
        prompt_eval_count=usage.prompt_tokens if usage else None,
        eval_count=usage.completion_tokens if usage else None,
        raw=completion,
    )


def _to_openai_messages(messages: List[Any]) -> List[Dict[str, Any]]:
    """Convert an Ollama-style conversation to the OpenAI format.

    OpenAI links each tool result to its call via `tool_call_id`, while the notebooks (like Ollama)
    only give the tool's name, so each `role="tool"` message is matched to the earliest unanswered
    call to that tool from the preceding assistant turn.
    """
    converted: List[Dict[str, Any]] = []
    pending: List[ToolCall] = []
    for msg in messages:
        if isinstance(msg, Message):
            entry: Dict[str, Any] = {"role": msg.role, "content": msg.content or ""}
            if msg.tool_calls:
                entry["tool_calls"] = [
                    tc.raw or {"id": tc.id, "type": "function",
                               "function": {"name": tc.function.name, "arguments": json.dumps(tc.function.arguments)}}
                    for tc in msg.tool_calls
                ]
                pending = list(msg.tool_calls)
            converted.append(entry)
            continue

        entry = dict(msg)
        tool_name = entry.pop("tool_name", None)
        if entry.get("role") == "tool" and "tool_call_id" not in entry and pending:
            match = next((tc for tc in pending if tc.function.name == tool_name), pending[0])
            pending.remove(match)
            entry["tool_call_id"] = match.id
        converted.append(entry)
    return converted
