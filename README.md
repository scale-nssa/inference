# inference

A thin, unified interface to Claude, Gemini, and GPT-OSS on Vertex AI, plus the
Stanford AI API Gateway. One `ask()` call across providers, a string-coercible
`Response`, and an escape hatch to the typed provider payload when you need it.

Vertex models need a Google Cloud project with the models enabled and Application
Default Credentials (`gcloud auth application-default login`). Stanford models need
a gateway key in `STANFORD_AI_API_KEY`.

## Install

```bash
uv add git+https://github.com/julian-bernado/inference
```

Or, from a sibling checkout, as an editable dependency:

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
model = llm.from_name("gemini_3_5_flash", temperature=0.2)
resp = model.ask(prompt, temperature=0.0)            # one-off override
```

Claude on Vertex requires `max_tokens`. Every Claude catalog entry sets one; override
it the same way:

```python
model = llm.from_name("claude_sonnet_4_6", max_tokens=4096)
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

Vertex Claude and Gemini catalog models support Cloud Storage–backed batch jobs.
(GPT-OSS on Vertex is wired up, but its jobs currently fail at Vertex's export stage;
run it synchronously.) Set `INFERENCE_BATCH_GCS_BUCKET` to a bucket or
`gs://bucket/prefix`; the package writes one subdirectory per job and leaves inputs
and outputs in place.

```python
from inference import BatchError, JobSucceeded, LLMOutput, Prompt, llm

prompts = [
    Prompt(system_prompt="Be terse.", model_prompt=f"Return item-{i}.")
    for i in range(1000)
]

model = llm.from_name("claude_sonnet_4_6", max_tokens=2048)
job_id = model.batch_submit(prompts, job_name="my-run")
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
model = llm.from_name("gemini_3_5_flash", temperature=0.0)
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
model = llm.from_name("gemini_3_5_flash", temperature=0.0)

job_id = model.batch_submit(prompts, job_name="my-run")
Path("my-run.job_id").write_text(job_id, encoding="utf-8")
```

```python
# collect.py
from pathlib import Path
from inference import BatchError, llm

job_id = Path("my-run.job_id").read_text(encoding="utf-8").strip()
model = llm.from_name("gemini_3_5_flash", temperature=0.0)

status = model.batch_status(job_id)
outputs = model.batch_collect(job_id)
```

`batch_collect(job_id)` recovers the original prompts from the Vertex job's GCS
input JSONL and reads the provider output JSONL from the job's output config.
Use the same provider/model family when collecting so the response parser
matches the output format. Do not delete the per-job GCS input/output directory
before collection.

### GPT-OSS on Vertex

`gpt_oss_120b_vertex` reaches Vertex's managed GPT-OSS service through its
OpenAI-compatible endpoint. Auth is a GCP access token from Application Default
Credentials (refreshed on demand), not an API key:

```python
from inference import Prompt, llm

model = llm.from_name("gpt_oss_120b_vertex")
response = model.ask(Prompt(system_prompt="Be terse.", model_prompt="Say hi."))
```

Reasoning goes out as a top-level `reasoning_effort` (`low` / `medium` / `high`).
`thinking_tokens` is always `None` here, because the endpoint already counts
reasoning inside `output_tokens`.

### Stanford AI API Gateway

Stanford's AI API Gateway (`https://aiapi-prod.stanford.edu/v1`) is
OpenAI-compatible. Set `STANFORD_AI_API_KEY` (the key from your gateway request;
it only reaches the models it was granted), then:

```python
from inference import Prompt, llm

model = llm.from_name("stanford_gpt_5_6_luna", reasoning_effort="high")
response = model.ask(Prompt(system_prompt="Be terse.", model_prompt="Say hi."))
```

Reasoning goes out as a top-level `reasoning_effort` and `max_tokens` is sent as
`max_completion_tokens`. The GPT-5.6 entries set `send_temperature: false`, so no
temperature is sent for them. The gateway has no batch endpoint.

**The gateway caches responses.** Sending the identical request again returns the
stored reply (same response id) instead of sampling the model again, at any
temperature. To draw several samples for one prompt, make each request differ, for
example with a unique id in the system prompt.

## The catalog

Models are defined as YAML files shipped at `inference/catalog/*.yaml`. Each entry
declares a `provider` (`claude_vertex`, `gemini_vertex`, `gpt_oss_vertex`, or
`stanford`) plus whatever kwargs the provider's pydantic class takes.

```yaml
nickname: claude_sonnet_4_5
name: Claude Sonnet 4.5
provider: claude_vertex
model_name: claude-sonnet-4-5@20250929
thinking: true
thinking_budget: 1024
ctx_length: 200000
max_tokens: 8192
project_id: gcp-nssa-ml
location: global
pricing:
  input: "$3.00 per 1M tokens"
  output: "$15.00 per 1M tokens"
source_url: https://cloud.google.com/model-garden/partner-models/pricing
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

### Model cards (the catalog is the source of truth for model metadata)

The catalog is also the single source of truth for model *cards* — intrinsic
metadata about a model (size, pricing, context window, provider, capabilities).
`llm.card(name)` resolves a name (same matching rules as `from_name`) and returns
a frozen `ModelCard` **as pure metadata**: it constructs no provider client, makes
no network calls, and needs no API credentials. `llm.cards()` returns all of them.

```python
from inference import llm

card = llm.card("stanford_deepseek_v3_2")
card.params_b         # -> 671    (total parameters, billions)
card.active_params_b  # -> 37     (active params for MoE models; None otherwise)
card.pricing          # -> {"input": "...", "output": "..."}
card.ctx_length       # -> 128000
card.reasoning        # -> mirrors the entry's `thinking` flag
card.logprobs         # -> True if this entry returns token-level logprobs
```

Card fields come from these catalog keys (all size/metadata keys are optional, so
partial coverage is fine and adding fields is additive):

| Card field         | Catalog key         | Notes |
|--------------------|---------------------|-------|
| `params_b`         | `params_b`          | Total parameters, in billions |
| `active_params_b`  | `active_params_b`   | Active params for MoE (e.g. DeepSeek-V3.2 → 37) |
| `reasoning`        | `thinking`          | Whether the model reasons/thinks |
| `logprobs`         | `supports_logprobs` | Entry returns token-level output logprobs |
| `verified_on`      | `verified`          | Last date the live call and price check both passed |
| `pricing`, `source_url`, `notes`, `provider`, `ctx_length`, `nickname`, `name`, `model_name` | same-named keys | |

Because cards need no credentials, other projects can depend on `inference` just to
read model metadata by nickname.

## Providers

- **`claude_vertex`** — Anthropic via Vertex AI. Supports extended thinking.
  Thinking uses the adaptive API with `effort` (default `medium`). Temperature is
  sent only with thinking off, and never for entries with `send_temperature: false`
  (Claude Opus 4.8 and the Claude 5 and 5.5 models reject it outright). Batch is
  50% off but runs in `us-east5`, which bills at Vertex's regional rate.
- **`gemini_vertex`** — Google GenAI via Vertex AI. Supports dynamic thinking.
  `thinking_level` defaults to `MEDIUM`. Thinking tokens are billed as output.
  Batch is 50% off.
- **`gpt_oss_vertex`** — GPT-OSS via Vertex AI's OpenAI-compatible endpoint.
  Authenticates with Application Default Credentials. `reasoning_effort` defaults
  to `medium`.
- **`stanford`** — Stanford AI API Gateway (OpenAI-compatible). Uses
  `STANFORD_AI_API_KEY`; top-level `reasoning_effort` (default `medium`),
  `max_completion_tokens`, and an optional `send_temperature: false`.

## Verification

`scripts/smoke_catalog.py` is the catalog's live check. For every entry it makes one
call at the entry's own settings, probes whether `temperature=0.0` is accepted, and
checks `pricing` against current rates. Vertex prices come from the Cloud Billing
Catalog API when it's enabled in your project, otherwise from Google's public pricing
pages; Stanford prices are measured by reading the billed cost of two calls back from
the gateway's usage API. Results go to
[`src/inference/catalog/VERIFIED.md`](src/inference/catalog/VERIFIED.md). Entries
whose call and price both pass get `verified: <date>` in their YAML, exposed as
`ModelCard.verified_on`.

```bash
gcloud auth application-default login
export STANFORD_AI_API_KEY=...
uv run scripts/smoke_catalog.py                 # all entries
uv run scripts/smoke_catalog.py --only claude_opus_5
```

## Development

```bash
uv sync                          # install, including the dev group (pytest)
uv run pytest                    # offline tests; SDK clients are mocked
uvx ruff check .                 # lint
```

To add a model, drop a YAML file into `src/inference/catalog/` (no Python changes),
then run `uv run scripts/smoke_catalog.py --only <nickname>` to verify it. How the
pieces fit together is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
