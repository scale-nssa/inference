# inference

A thin, unified interface to Claude (via Vertex), Gemini (via Vertex),
OpenRouter, and any OpenAI-compatible local model served by LM Studio. One
`ask()` call across providers, a string-coercible `Response`, and an escape
hatch to the typed provider payload when you need it.

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

Defaults are `temperature=0.8`, with no package-level `max_tokens` cap. Set
either on the model, or pass to a specific call:

```python
model = llm.from_name("gemini_3_flash", temperature=0.2)
resp = model.ask(prompt, temperature=0.0)            # one-off override
```

Claude Vertex requires `max_tokens`; set it explicitly for Claude models:

```python
model = llm.from_name("claude_sonnet_4_6", temperature=0.0, max_tokens=8192)
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

### Vertex batch prediction

Vertex Claude and Vertex Gemini catalog models support Cloud Storage backed
batch jobs. Set `INFERENCE_BATCH_GCS_BUCKET` to a bucket or `gs://bucket/prefix`;
the package writes one per-job subdirectory and leaves inputs/outputs in place.

```python
from inference import BatchError, JobSucceeded, LLMOutput, Prompt, llm

prompts = [
    Prompt(system_prompt="Be terse.", model_prompt=f"Return item-{i}.")
    for i in range(1000)
]

model = llm.from_name("claude_sonnet_4_6", temperature=0.0, max_tokens=2048)
job_id = model.batch_submit(prompts, job_name="annotation-run")
status = model.batch_status(job_id)
outputs = model.batch_collect(job_id)

assert isinstance(status, JobSucceeded) or status.state == "running"
for result in outputs:
    if isinstance(result, BatchError):
        print(result.index, result.status)  # per-request failure; resubmit later
    else:
        assert isinstance(result, LLMOutput)
        print(result.text, result.tokens)
```

For submit + poll + collect in one call:

```python
model = llm.from_name("gemini_3_flash", temperature=0.0)
outputs = model.batch(prompts, poll_interval=30.0)
```

For long-running jobs, use the split API and persist the returned `job_id`.
Vertex owns the job after submission, so your local process and computer do not
need to stay online. Later, recreate the same catalog model and collect by job
ID:

```python
# submit.py
from pathlib import Path
from inference import Prompt, llm

prompts = [Prompt(system_prompt="Be terse.", model_prompt=f"Return item-{i}.") for i in range(5)]
model = llm.from_name("gemini_3_flash", temperature=0.0)

job_id = model.batch_submit(prompts, job_name="annotation-run")
Path("annotation-run.job_id").write_text(job_id, encoding="utf-8")
```

```python
# collect.py
from pathlib import Path
from inference import BatchError, llm

job_id = Path("annotation-run.job_id").read_text(encoding="utf-8").strip()
model = llm.from_name("gemini_3_flash", temperature=0.0)

status = model.batch_status(job_id)
outputs = model.batch_collect(job_id)
```

`batch_collect(job_id)` recovers the original prompts from the Vertex job's GCS
input JSONL and reads the provider output JSONL from the job's output config.
Use the same provider/model family when collecting so the response parser
matches the output format. Do not delete the per-job GCS input/output directory
before collection.

Batch is Vertex-only in this release. `model.call(...)` and `model.ask(...)`
continue to run synchronous single-prompt inference unchanged.

### OpenRouter

OpenRouter uses the existing OpenAI SDK dependency pointed at OpenRouter's
OpenAI-compatible endpoint. Set `OPENROUTER_API_KEY`, then select a catalog
model or add your own OpenRouter catalog YAML:

```python
from inference import Prompt, llm

model = llm.from_name("openrouter_auto")
resp = model.ask(Prompt(system_prompt="Be terse.", model_prompt="Say hi."))
```

Optional attribution headers can be set with `OPENROUTER_HTTP_REFERER` and
`OPENROUTER_APP_TITLE`, or via constructor overrides:

```python
model = llm.from_name(
    "openrouter_auto",
    model_name="anthropic/claude-sonnet-4.5",
    http_referer="https://example.com",
    app_title="my-app",
)
```

## The catalog

Models are defined as YAML files shipped at `inference/catalog/*.yaml`. Each entry
declares a `provider` (`claude_vertex`, `gemini_vertex`, `openrouter`, or
`lm_studio`) plus whatever kwargs the provider's pydantic class takes.

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
- **`openrouter`** — OpenRouter through its OpenAI-compatible chat completions
  endpoint. Uses `OPENROUTER_API_KEY`; optional `extra_body` passes
  OpenRouter-specific routing/provider options through to the request.
- **`lm_studio`** — any OpenAI-compatible LM Studio model. Auto-loads the model
  at construction (`auto_load=False` to disable). Two knobs:
  - `system_role`: `"system"` (default) or `"developer"` (gpt-oss Harmony)
  - `thinking_style`: `"none"`, `"reasoning_attr"` (gpt-oss), or `"think_tag"` (qwen `<think>...</think>`)
