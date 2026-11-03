# Architecture

How `inference` is put together, for contributors. For usage, see the
[README](../README.md).

## Request flow

Every provider follows the same path:

```
llm.from_name(name) → resolve a catalog YAML entry → construct a Model subclass
model.ask() / model.call() → Model.call() builds InferenceSettings → provider _call() → LLMOutput
```

- **`Model`** (`providers/base.py`) is a frozen pydantic base. It holds the shared
  fields (`nickname`, `model_name`, `thinking`, `ctx_length`, `temperature`,
  `max_tokens`, `logprobs`) and implements `ask`, `call`, and the `batch_*` methods.
  Each provider subclass adds its own fields and implements only
  `_call(prompt, settings) -> LLMOutput`.
- **`InferenceSettings`** (`settings.py`) is the resolved per-call configuration.
  `Model._settings()` merges the model's defaults with per-call overrides.
  `temperature` defaults to `0.8`; there is no package-wide `max_tokens` default.
- **Responses** (`response.py`): `LLMOutput` is the full result (prompt, text,
  thinking, `TokenStats`, typed provider payload); `Response` is the lean version
  returned by `ask()`. Both convert to their text with `str()`.
- **Provider payloads** (`provider_responses/`) are typed pydantic models of each
  provider's raw response (`AnthropicMessageResponse`, `GeminiGenerateResponse`,
  `OpenAICompletionResponse`). Providers serialize the SDK object with
  `_dump.dump_provider_response()` and re-validate it, so nothing is lost.

## The catalog

Models are data, not code: one YAML file per model in `src/inference/catalog/`.
Each entry names a `provider` (`claude_vertex`, `gemini_vertex`, `gpt_oss_vertex`,
`stanford`); every other key is passed to that provider's constructor.
`PROVIDERS` in `providers/__init__.py` maps provider names to classes. Adding a model
means adding a YAML file.

Name resolution (`llm._resolve`) tries, in order: exact nickname, exact
`model_name`, normalized match (lowercase, alphanumerics only, `@version` dropped),
then a unique normalized substring. Ambiguity raises `AmbiguousModelError`; no match
raises `UnknownModelError`. The catalog is cached; `llm.reload_catalog()` re-reads
it. `INFERENCE_CATALOG_PATH` adds an external directory whose entries override
built-ins with the same nickname.

### Model cards

`llm.card(name)` / `llm.cards()` return `ModelCard`s (`model_card.py`): metadata read
straight from the catalog, with no provider construction, network access, or
credentials. Two keys are renamed: `reasoning` mirrors `thinking`, and `logprobs`
mirrors `supports_logprobs` (kept separate from `Model.logprobs`, which controls
whether a request asks for logprobs). `verified_on` mirrors `verified` (see
[Verification](#verification)). Metadata-only keys such as `pricing`, `params_b`,
and `notes` are ignored by the provider constructors. New card fields should be
optional.

## Providers

- **`claude_vertex`** (`providers/claude_vertex.py`): Anthropic SDK against Vertex.
  Vertex requires `max_tokens`. `thinking_api: adaptive` sends
  `thinking: {type: adaptive}` plus `output_config.effort`; `thinking_api: enabled`
  (the class default, for models without adaptive thinking) sends a token budget,
  and `max_tokens` must exceed `thinking_budget`. Temperature is sent only when
  thinking is off and `send_temperature` is true.
- **`gemini_vertex`** (`providers/gemini_vertex.py`): Google GenAI SDK against
  Vertex. `thinking_level` defaults to `MEDIUM`.
- **`gpt_oss_vertex`** and **`stanford`** both subclass `OpenAICompatModel`
  (`providers/openai_compat.py`), which builds OpenAI chat-completions requests,
  parses responses (including logprobs), and keeps one pooled `OpenAI` client per
  endpoint and credential. Subclasses supply `_client()` and adjust requests through
  `_request_kwargs` / `_request_extra_body`. Both send reasoning as a top-level
  `reasoning_effort` and support `send_temperature: false`.
  - **`gpt_oss_vertex`** authenticates with a refreshed GCP access token, and derives
    `base_url` from `project_id` and `location`. It reports `thinking_tokens` as
    `None` because the endpoint already counts reasoning in `output_tokens`.
  - **`stanford`** uses `STANFORD_AI_API_KEY`, sends `max_tokens` as
    `max_completion_tokens`, and has no batch endpoint.

All entries default to medium thinking effort where the model supports it, both as
the provider class default and explicitly in each YAML.

## Batch

`batch.py` runs Cloud Storage–backed Vertex batch prediction for `claude_vertex`,
`gemini_vertex`, and `gpt_oss_vertex`; `Model.batch_*` delegates to it. The API is
split into `batch_submit` → `batch_status` → `batch_collect`, plus an all-in-one
`batch`. Requests are correlated by `custom_id` (`request-{index:012d}`), and
per-request failures come back as `BatchError` items instead of raising. Result and
status types live in `batch_types.py`.

Set `INFERENCE_BATCH_GCS_BUCKET` to a bucket or `gs://bucket/prefix`. Vertex owns the
job after submission, so persist the `job_id` and recreate the same model to
collect; the output parser is chosen by provider. Claude batch jobs run in
`us-east5` (override with `INFERENCE_BATCH_VERTEX_LOCATION`) and GPT-OSS jobs in
`us-central1`. Keep `batch._format_request` in step with each provider's `_call`.

## Verification

`scripts/smoke_catalog.py` checks every catalog entry live:

- **call**: one request at the entry's own settings;
- **temperature**: whether the model accepts `temperature=0`. Accepted doesn't prove
  it's applied; raw answers at 0 and at a high temperature are kept in
  `verification.json` for reference, but black-box sampling can't reliably tell
  whether a thinking model applies temperature, so no verdict is recorded;
- **cache**: whether two identical requests come back with the same response id,
  which means the endpoint returns cached responses (the Stanford gateway does);
- **price**: the entry's `pricing` against current rates, via
  `scripts/catalog_prices.py`. Vertex prices come from the Cloud Billing Catalog API
  when it's enabled, otherwise from Google's pricing pages (only the provider's own
  section, standard tier, global row). Stanford publishes no machine-readable rates,
  so two lopsided calls are made and their billed cost is read back from the
  gateway's usage API, then solved for the input and output rate.

Results are merged into `catalog/verification.json` and rendered to
`catalog/VERIFIED.md`. An entry whose call and price check both pass gets
`verified: <date>` in its YAML; an entry that fails either loses it.

## Tests

`uv run pytest` is fully offline: tests monkeypatch the SDK clients (`OpenAI`,
`AnthropicVertex`, `genai.Client`, the GCS and Vertex batch clients) and assert on
the exact request payloads. Changing how a provider builds a request means
updating those assertions. The two live (billable) batch tests are skipped unless
`INFERENCE_RUN_BATCH_E2E=1` is set.

## Conventions

- User-facing types (`Prompt`, `Response`, `LLMOutput`, `Model` subclasses, batch
  types) are frozen pydantic models.
- `Prompt` is `system_prompt: str | None` plus `model_prompt: str`.
- The package uses a `src/` layout; the catalog directory is force-included in the
  wheel (`pyproject.toml`).
