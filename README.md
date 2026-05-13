# inference

A thin, unified interface to Claude (via Vertex), Gemini (via Vertex), and any
OpenAI-compatible local model served by LM Studio. One `ask()` call across
providers, a string-coercible `Response`, and an escape hatch to the typed
provider payload when you need it.

## Install

From a sibling project, add as an editable dependency:

```bash
uv add --editable ../inference
```

## Usage

```python
from inference import llm, Prompt

model = llm.from_name("claude-sonnet-4-5")           # name-pattern matching against the catalog
# model = llm.from_name("claude_sonnet_4_5")          # exact nickname
# model = llm.from_name("sonnet-4-5")                 # unique substring also works

prompt = Prompt(
    system_prompt="Be terse.",
    model_prompt="What's the capital of Italy? Respond with just the name.",
)

resp = model.ask(prompt)
print(resp)                # "Rome"   (Response.__str__ returns the text)
resp.text                  # "Rome"
resp.thinking              # "Hm, I think this is Rome..." or None
resp.input_tokens          # 12
resp.output_tokens         # 3
resp.thinking_tokens       # 47 or None
```

### Constructor settings (overridable per-call)

Defaults are `temperature=0.8`, `max_tokens=8192`. Set on the model, or pass to a
specific call:

```python
model = llm.from_name("claude-sonnet-4-5", temperature=0.2, max_tokens=2048)
resp = model.ask(prompt, temperature=0.0)            # one-off override
```

### Typed provider response

When you need the lossless provider payload (tool calls, safety ratings, raw
finish reasons, etc.) use `llm.call` instead of `model.ask`:

```python
out = llm.call(model, prompt)
out.text                   # "Rome"
out.thinking
out.tokens                 # TokenStats
out.prompt                 # the Prompt that was sent
out.provider_response      # AnthropicMessageResponse | OpenAICompletionResponse | GeminiGenerateResponse
```

## The catalog

Models are defined as YAML files shipped at `inference/catalog/*.yaml`. Each entry
declares a `provider` (`claude_vertex`, `gemini_vertex`, or `lm_studio`) plus
whatever kwargs the provider's pydantic class takes.

```yaml
nickname: claude_sonnet_4_5
name: Claude Sonnet 4.5
provider: claude_vertex
model_name: claude-sonnet-4-5@20250929
thinking: true
ctx_length: 200000
project_id: gcp-nssa-ml
location: global
```

Extend the catalog from outside the package with the `INFERENCE_CATALOG_PATH`
environment variable (entries there override built-in ones with the same nickname).

`llm.from_name(s)` matches `s` against the catalog in this order:
1. Exact `nickname`
2. Exact `model_name`
3. Normalized match (lowercase, strip non-alphanumeric, drop `@version`)
4. Unique normalized-substring match

Ambiguous matches raise `AmbiguousModelError`; no match raises `UnknownModelError`.

`llm.available()` lists every nickname.

## Providers

- **`claude_vertex`** — Anthropic via Vertex AI. Supports extended thinking.
- **`gemini_vertex`** — Google GenAI via Vertex AI. Supports dynamic thinking.
- **`lm_studio`** — any OpenAI-compatible LM Studio model. Auto-loads the model
  at construction (`auto_load=False` to disable). Two knobs:
  - `system_role`: `"system"` (default) or `"developer"` (gpt-oss Harmony)
  - `thinking_style`: `"none"`, `"reasoning_attr"` (gpt-oss), or `"think_tag"` (qwen `<think>...</think>`)
