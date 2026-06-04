from __future__ import annotations

import json
import os
import time

import pytest

from inference import BatchError, LLMOutput, Prompt, llm
from inference.batch import (
    batch_submit,
    format_input_jsonl,
    parse_output_jsonl,
)
from inference.providers.claude_vertex import ClaudeVertexModel
from inference.providers.gemini_vertex import GeminiVertexModel


def _jsonl_rows(payload: bytes) -> list[dict]:
    return [json.loads(line) for line in payload.decode("utf-8").splitlines()]


def test_claude_batch_input_jsonl_matches_vertex_schema() -> None:
    model = ClaudeVertexModel(
        nickname="claude",
        name="Claude",
        model_name="claude-sonnet-4-6",
        thinking=True,
        ctx_length=200000,
        max_tokens=8192,
    )
    prompts = [
        Prompt(system_prompt="Be terse.", model_prompt="Say alpha."),
        Prompt(system_prompt=None, model_prompt="Say beta."),
    ]

    rows = _jsonl_rows(
        format_input_jsonl(
            provider="claude_vertex",
            prompts=prompts,
            model=model,
            settings=model._settings(None, None),
        )
    )

    assert rows[0]["custom_id"] == "request-000000000000"
    assert rows[0]["request"]["anthropic_version"] == "vertex-2023-10-16"
    assert rows[0]["request"]["system"] == "Be terse."
    assert rows[0]["request"]["messages"] == [
        {"role": "user", "content": "Say alpha."}
    ]
    assert rows[0]["request"]["thinking"] == {
        "type": "enabled",
        "budget_tokens": 1024,
    }
    assert rows[1]["custom_id"] == "request-000000000001"


def test_claude_thinking_budget_must_be_less_than_max_tokens() -> None:
    model = ClaudeVertexModel(
        nickname="claude",
        name="Claude",
        model_name="claude-sonnet-4-6",
        thinking=True,
        ctx_length=200000,
        thinking_budget=1024,
        max_tokens=1024,
    )

    with pytest.raises(ValueError, match="max_tokens=1024, thinking_budget=1024"):
        format_input_jsonl(
            provider="claude_vertex",
            prompts=[Prompt(system_prompt=None, model_prompt="hello")],
            model=model,
            settings=model._settings(None, None),
        )


def test_claude_batch_requires_max_tokens() -> None:
    model = ClaudeVertexModel(
        nickname="claude",
        name="Claude",
        model_name="claude-sonnet-4-6",
        thinking=True,
        ctx_length=200000,
        max_tokens=None,
    )

    with pytest.raises(ValueError, match="Claude Vertex batch requests require max_tokens"):
        format_input_jsonl(
            provider="claude_vertex",
            prompts=[Prompt(system_prompt=None, model_prompt="hello")],
            model=model,
            settings=model._settings(None, None),
        )


def test_gemini_batch_input_jsonl_matches_vertex_schema() -> None:
    model = GeminiVertexModel(
        nickname="gemini",
        name="Gemini",
        model_name="gemini-3-flash-preview",
        thinking=True,
        ctx_length=1000000,
    )
    prompt = Prompt(system_prompt="Be terse.", model_prompt="Say alpha.")

    rows = _jsonl_rows(
        format_input_jsonl(
            provider="gemini_vertex",
            prompts=[prompt],
            model=model,
            settings=model._settings(None, None),
        )
    )

    assert rows == [
        {
            "custom_id": "request-000000000000",
            "request": {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": "Say alpha."}],
                    }
                ],
                "generationConfig": {
                    "temperature": 0.8,
                    "thinkingConfig": {"includeThoughts": True},
                },
                "systemInstruction": {
                    "role": "system",
                    "parts": [{"text": "Be terse."}],
                },
            },
        }
    ]


def test_gemini_batch_includes_max_output_tokens_when_set() -> None:
    model = GeminiVertexModel(
        nickname="gemini",
        name="Gemini",
        model_name="gemini-3-flash-preview",
        thinking=False,
        ctx_length=1000000,
        max_tokens=2048,
    )

    rows = _jsonl_rows(
        format_input_jsonl(
            provider="gemini_vertex",
            prompts=[Prompt(system_prompt=None, model_prompt="Say alpha.")],
            model=model,
            settings=model._settings(None, None),
        )
    )

    assert rows[0]["request"]["generationConfig"] == {
        "temperature": 0.8,
        "maxOutputTokens": 2048,
    }


def test_claude_batch_output_jsonl_is_ordered_by_custom_id() -> None:
    prompts = [
        Prompt(system_prompt=None, model_prompt="first"),
        Prompt(system_prompt=None, model_prompt="second"),
    ]
    lines = [
        json.dumps(
            {
                "custom_id": "request-000000000001",
                "request": {"system": None, "messages": [{"content": "second"}]},
                "status": "",
                "response": {
                    "id": "msg_2",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-6",
                    "content": [{"type": "text", "text": "two"}],
                    "usage": {"input_tokens": 11, "output_tokens": 2},
                },
            }
        ),
        json.dumps(
            {
                "custom_id": "request-000000000000",
                "request": {"system": None, "messages": [{"content": "first"}]},
                "status": "",
                "response": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-6",
                    "content": [
                        {"type": "thinking", "thinking": "thinking"},
                        {"type": "text", "text": "one"},
                    ],
                    "usage": {"input_tokens": 10, "output_tokens": 1},
                },
            }
        ),
    ]

    outputs = parse_output_jsonl(
        provider="claude_vertex",
        prompts=prompts,
        lines=lines,
    )

    assert [output.text for output in outputs if isinstance(output, LLMOutput)] == [
        "one",
        "two",
    ]
    assert isinstance(outputs[0], LLMOutput)
    assert outputs[0].thinking == "thinking"
    assert outputs[0].tokens.input_tokens == 10


def test_gemini_batch_output_jsonl_is_ordered_and_parses_camel_case() -> None:
    prompts = [
        Prompt(system_prompt="s", model_prompt="first"),
        Prompt(system_prompt="s", model_prompt="second"),
    ]
    lines = [
        json.dumps(
            {
                "custom_id": "request-000000000001",
                "request": {
                    "systemInstruction": {"parts": [{"text": "s"}]},
                    "contents": [{"parts": [{"text": "second"}]}],
                },
                "status": "",
                "response": {
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"text": "two"}],
                                "role": "model",
                            },
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 4,
                        "candidatesTokenCount": 5,
                        "thoughtsTokenCount": 6,
                    },
                },
            }
        ),
        json.dumps(
            {
                "custom_id": "request-000000000000",
                "request": {
                    "systemInstruction": {"parts": [{"text": "s"}]},
                    "contents": [{"parts": [{"text": "first"}]}],
                },
                "status": "",
                "response": {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"text": "thought", "thought": True},
                                    {"text": "one"},
                                ],
                                "role": "model",
                            },
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 1,
                        "candidatesTokenCount": 2,
                        "thoughtsTokenCount": 3,
                    },
                },
            }
        ),
    ]

    outputs = parse_output_jsonl(
        provider="gemini_vertex",
        prompts=prompts,
        lines=lines,
    )

    assert [output.text for output in outputs if isinstance(output, LLMOutput)] == [
        "one",
        "two",
    ]
    assert isinstance(outputs[0], LLMOutput)
    assert outputs[0].thinking == "thought"
    assert outputs[0].tokens.thinking_tokens == 3
    assert outputs[0].provider_response.usage_metadata.prompt_token_count == 1


def test_batch_output_preserves_per_request_failures() -> None:
    prompts = [Prompt(system_prompt=None, model_prompt="bad")]
    outputs = parse_output_jsonl(
        provider="gemini_vertex",
        prompts=prompts,
        lines=[
            json.dumps(
                {
                    "custom_id": "request-000000000000",
                    "request": {"contents": [{"parts": [{"text": "bad"}]}]},
                    "status": "Bad Request: invalid role",
                    "response": {},
                }
            )
        ],
    )

    assert outputs == [
        BatchError(
            prompt=prompts[0],
            index=0,
            request_id="request-000000000000",
            status="Bad Request: invalid role",
            response={},
        )
    ]


def test_batch_submit_uses_aiplatform_and_storage_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uploaded: dict[str, object] = {}
    created: dict[str, object] = {}

    class FakeBlob:
        def __init__(self, name: str) -> None:
            self.name = name

        def upload_from_string(self, data: bytes, content_type: str) -> None:
            uploaded["name"] = self.name
            uploaded["data"] = data
            uploaded["content_type"] = content_type

    class FakeBucket:
        def __init__(self, name: str) -> None:
            self.name = name

        def blob(self, name: str) -> FakeBlob:
            return FakeBlob(name)

    class FakeStorageClient:
        def __init__(self, project: str | None = None) -> None:
            uploaded["project"] = project

        def bucket(self, name: str) -> FakeBucket:
            uploaded["bucket"] = name
            return FakeBucket(name)

    class FakeJobClient:
        def __init__(self, client_options: dict[str, str]) -> None:
            created["client_options"] = client_options

        def create_batch_prediction_job(self, parent: str, batch_prediction_job):
            created["parent"] = parent
            created["job"] = batch_prediction_job

            class Response:
                name = "projects/p/locations/us-east5/batchPredictionJobs/123"

            return Response()

    monkeypatch.setenv("INFERENCE_BATCH_GCS_BUCKET", "gs://bucket/prefix")
    monkeypatch.setattr("inference.batch.storage.Client", FakeStorageClient)
    monkeypatch.setattr("inference.batch.aiplatform_v1.JobServiceClient", FakeJobClient)
    monkeypatch.setattr("inference.batch.uuid.uuid4", lambda: type("U", (), {"hex": "abc123def456"})())
    model = ClaudeVertexModel(
        nickname="claude",
        name="Claude",
        model_name="claude-sonnet-4-6",
        thinking=False,
        ctx_length=200000,
        project_id="p",
        location="global",
        max_tokens=8192,
    )

    job_id = batch_submit(
        model,
        [Prompt(system_prompt=None, model_prompt="hello")],
        job_name="test job",
    )

    assert job_id == "projects/p/locations/us-east5/batchPredictionJobs/123"
    assert uploaded["bucket"] == "bucket"
    assert uploaded["name"] == "prefix/test-job-abc123def456/input/requests.jsonl"
    assert uploaded["content_type"] == "application/jsonl"
    assert created["client_options"] == {"api_endpoint": "us-east5-aiplatform.googleapis.com"}
    assert created["parent"] == "projects/p/locations/us-east5"
    created_job = created["job"]
    assert created_job.model == "publishers/anthropic/models/claude-sonnet-4-6"
    assert created_job.input_config.gcs_source.uris == [
        "gs://bucket/prefix/test-job-abc123def456/input/requests.jsonl"
    ]
    assert (
        created_job.output_config.gcs_destination.output_uri_prefix
        == "gs://bucket/prefix/test-job-abc123def456/output"
    )


@pytest.mark.skipif(
    os.environ.get("INFERENCE_RUN_BATCH_E2E") != "1",
    reason="set INFERENCE_RUN_BATCH_E2E=1 to run billable Vertex batch smoke tests",
)
@pytest.mark.parametrize("model_name", ["claude_sonnet_4_6", "gemini_3_flash"])
def test_vertex_batch_e2e_smoke(model_name: str) -> None:
    prompts = [
        Prompt(
            system_prompt="Answer with exactly one short line.",
            model_prompt=f"Return the word item-{i}.",
        )
        for i in range(5)
    ]
    model = llm.from_name(model_name, temperature=0.0, max_tokens=2048)
    start = time.monotonic()
    outputs = model.batch(
        prompts,
        job_name=f"pytest-{model_name}",
        poll_interval=30.0,
    )
    elapsed = time.monotonic() - start

    assert len(outputs) == len(prompts)
    assert [output.prompt for output in outputs] == prompts
    assert any(isinstance(output, LLMOutput) and output.text.strip() for output in outputs)
    print(
        f"{model_name} batch smoke completed in {elapsed:.1f}s: "
        f"{[str(output)[:80] for output in outputs]}"
    )
