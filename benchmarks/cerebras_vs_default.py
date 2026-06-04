"""Price/latency/throughput comparison: gpt-oss-120b on OpenRouter.

Compares two routing configs for `openai/gpt-oss-120b`:
  * default  -- no provider pinned; OpenRouter picks.
  * cerebras -- pinned to the Cerebras provider only.

Generates a reproducible set of mixed, nontrivial prompts, fires them at each
config through a thread pool, and reports per-request latency, throughput
(output tokens/sec), and actual cost (from OpenRouter's `usage.cost`).

Requires OPENROUTER_API_KEY in the environment. Run it yourself:

    uv run python benchmarks/cerebras_vs_default.py --n 100 --workers 8

Add `--config default` or `--config cerebras` to run just one side.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from inference import Prompt, llm

MODEL = "openrouter_gpt_oss_120b"

CONFIGS: dict[str, dict] = {
    # No provider pinned: OpenRouter routes wherever it likes.
    "default": {"usage": {"include": True}},
    # Force Cerebras only (no fallbacks to other providers).
    "cerebras": {
        "provider": {"only": ["cerebras"], "allow_fallbacks": False},
        "usage": {"include": True},
    },
}


# --------------------------------------------------------------------------
# Prompt generation -- mixed domains, all sized to demand substantial output.
# --------------------------------------------------------------------------

_TOPICS = [
    "distributed consensus", "the carbon cycle", "monetary policy",
    "CRISPR gene editing", "the Roman aqueduct system", "reinforcement learning",
    "plate tectonics", "the printing press", "supply chain logistics",
    "immune system response", "quantum entanglement", "the French Revolution",
    "neural network backpropagation", "ocean thermohaline circulation",
    "the transistor", "antibiotic resistance", "general relativity",
    "the stock market", "photosynthesis", "the internet's routing layer",
]

_SYSTEMS = [
    "a rate limiter for a public API", "a URL shortener", "a job queue with retries",
    "an LRU cache", "a real-time leaderboard", "a file deduplication service",
    "a multiplayer game lobby", "a notification fan-out system",
    "a feature-flag service", "a time-series metrics store",
]

_LANGS = ["Python", "Rust", "Go", "TypeScript"]

_SCENARIOS = [
    "a city deciding whether to build a new subway line",
    "a startup choosing between two pricing models",
    "a hospital triaging patients during a staffing shortage",
    "a government weighing a carbon tax against subsidies",
    "a team migrating a monolith to microservices",
    "a farmer deciding which crops to plant given a drought forecast",
    "a publisher choosing between print and digital-first strategy",
    "a national park managing an overpopulation of deer",
]


def _mixed_prompts(n: int, seed: int = 1234) -> list[Prompt]:
    """Build `n` reproducible prompts across explanation/code/analysis/writing."""
    rng = random.Random(seed)
    system = (
        "You are a meticulous expert. Answer thoroughly and precisely, "
        "using structure (sections, lists, or steps) where it helps."
    )
    builders = [
        lambda: (
            f"Explain {rng.choice(_TOPICS)} from first principles. Cover the "
            "core mechanism, why it matters, a concrete worked example, two "
            "common misconceptions, and where the simplified model breaks down. "
            "Aim for depth a graduate student would respect."
        ),
        lambda: (
            f"Design and implement {rng.choice(_SYSTEMS)} in "
            f"{rng.choice(_LANGS)}. Give the data model, the core algorithm, "
            "complete commented code, the time/space complexity, and how it "
            "behaves under concurrency and failure."
        ),
        lambda: (
            f"Analyze the tradeoffs facing {rng.choice(_SCENARIOS)}. Lay out "
            "the key options, the stakeholders, second-order effects, the "
            "strongest argument on each side, and a reasoned recommendation "
            "with the conditions under which you'd reverse it."
        ),
        lambda: (
            f"Compare and contrast {rng.choice(_TOPICS)} and "
            f"{rng.choice(_TOPICS)}. Identify the deep structural similarity, "
            "three concrete differences, and what one field could borrow from "
            "the other. Use a table where useful."
        ),
        lambda: (
            f"Write a tight, vivid 400-600 word piece that uses "
            f"{rng.choice(_TOPICS)} as an extended metaphor for "
            f"{rng.choice(_SCENARIOS)}. Then add a short paragraph unpacking "
            "how the metaphor holds up and where it strains."
        ),
    ]
    prompts: list[Prompt] = []
    for i in range(n):
        builder = builders[i % len(builders)]
        prompts.append(Prompt(system_prompt=system, model_prompt=builder()))
    return prompts


# --------------------------------------------------------------------------
# Benchmark
# --------------------------------------------------------------------------


@dataclass
class Result:
    idx: int
    ok: bool
    latency_s: float
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cost_usd: float | None = None
    error: str | None = None

    @property
    def tok_per_s(self) -> float:
        return self.output_tokens / self.latency_s if self.latency_s > 0 else 0.0


def _run_one(model, prompt: Prompt, idx: int) -> Result:
    start = time.perf_counter()
    try:
        out = llm.call(model, prompt)
    except Exception as e:  # noqa: BLE001 -- record provider/network failures
        return Result(idx=idx, ok=False, latency_s=time.perf_counter() - start,
                      error=f"{type(e).__name__}: {e}")
    latency = time.perf_counter() - start
    usage = out.provider_response.usage
    return Result(
        idx=idx,
        ok=True,
        latency_s=latency,
        input_tokens=out.tokens.input_tokens,
        output_tokens=out.tokens.output_tokens,
        thinking_tokens=out.tokens.thinking_tokens or 0,
        cost_usd=usage.cost if usage is not None else None,
    )


def _bench(config_name: str, prompts: list[Prompt], workers: int) -> dict:
    model = llm.from_name(MODEL, extra_body=CONFIGS[config_name])
    results: list[Result] = []
    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(_run_one, model, p, i): i
            for i, p in enumerate(prompts)
        }
        done = 0
        for fut in as_completed(futs):
            results.append(fut.result())
            done += 1
            print(f"  [{config_name}] {done}/{len(prompts)}", end="\r", flush=True)
    wall = time.perf_counter() - wall_start
    print()
    results.sort(key=lambda r: r.idx)
    return _summarize(config_name, results, wall)


def _summarize(config_name: str, results: list[Result], wall_s: float) -> dict:
    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    lat = [r.latency_s for r in ok]
    out_tok = [r.output_tokens for r in ok]
    in_tok = [r.input_tokens for r in ok]
    per_req_tps = [r.tok_per_s for r in ok]
    costs = [r.cost_usd for r in ok if r.cost_usd is not None]

    def pct(xs: list[float], p: float) -> float:
        if not xs:
            return 0.0
        s = sorted(xs)
        k = max(0, min(len(s) - 1, int(round((p / 100) * (len(s) - 1)))))
        return s[k]

    total_out = sum(out_tok)
    return {
        "config": config_name,
        "n_total": len(results),
        "n_ok": len(ok),
        "n_failed": len(failed),
        "errors": [f"#{r.idx}: {r.error}" for r in failed][:10],
        "wall_clock_s": round(wall_s, 2),
        "latency_s": {
            "mean": round(statistics.mean(lat), 2) if lat else 0,
            "median": round(statistics.median(lat), 2) if lat else 0,
            "p95": round(pct(lat, 95), 2),
            "min": round(min(lat), 2) if lat else 0,
            "max": round(max(lat), 2) if lat else 0,
        },
        "tokens": {
            "input_total": sum(in_tok),
            "output_total": total_out,
            "output_mean_per_req": round(statistics.mean(out_tok), 1) if out_tok else 0,
        },
        "throughput": {
            # Single-request decode speed (mean over requests).
            "per_request_tok_per_s_mean": round(statistics.mean(per_req_tps), 1)
            if per_req_tps else 0,
            "per_request_tok_per_s_median": round(statistics.median(per_req_tps), 1)
            if per_req_tps else 0,
            # Aggregate system throughput across the whole concurrent run.
            "aggregate_output_tok_per_s": round(total_out / wall_s, 1) if wall_s else 0,
        },
        "cost_usd": {
            "total": round(sum(costs), 6) if costs else None,
            "mean_per_req": round(statistics.mean(costs), 6) if costs else None,
            "per_1k_requests_est": round(statistics.mean(costs) * 1000, 2)
            if costs else None,
            "reported_for_n": len(costs),
        },
    }


def _print_report(summaries: list[dict]) -> None:
    print("\n" + "=" * 72)
    print("RESULTS: gpt-oss-120b  (default routing  vs  Cerebras)")
    print("=" * 72)
    rows = [
        ("requests ok/total", lambda s: f"{s['n_ok']}/{s['n_total']}"),
        ("wall clock (s)", lambda s: s["wall_clock_s"]),
        ("latency mean (s)", lambda s: s["latency_s"]["mean"]),
        ("latency median (s)", lambda s: s["latency_s"]["median"]),
        ("latency p95 (s)", lambda s: s["latency_s"]["p95"]),
        ("out tok / req (mean)", lambda s: s["tokens"]["output_mean_per_req"]),
        ("per-req tok/s (mean)", lambda s: s["throughput"]["per_request_tok_per_s_mean"]),
        ("aggregate tok/s", lambda s: s["throughput"]["aggregate_output_tok_per_s"]),
        ("total cost ($)", lambda s: s["cost_usd"]["total"]),
        ("cost / req ($)", lambda s: s["cost_usd"]["mean_per_req"]),
        ("est $ / 1k reqs", lambda s: s["cost_usd"]["per_1k_requests_est"]),
    ]
    names = [s["config"] for s in summaries]
    w = 22
    print(f"{'metric':<24}" + "".join(f"{n:>{w}}" for n in names))
    print("-" * (24 + w * len(names)))
    for label, fn in rows:
        print(f"{label:<24}" + "".join(f"{str(fn(s)):>{w}}" for s in summaries))
    for s in summaries:
        if s["n_failed"]:
            print(f"\n{s['config']} errors ({s['n_failed']}):")
            for e in s["errors"]:
                print(f"  {e}")
    print("=" * 72)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="number of questions")
    ap.add_argument("--workers", type=int, default=8, help="concurrent requests")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument(
        "--config",
        choices=["default", "cerebras", "both"],
        default="both",
    )
    ap.add_argument("--json-out", type=str, default=None, help="write raw summary JSON")
    args = ap.parse_args()

    prompts = _mixed_prompts(args.n, seed=args.seed)
    print(f"Generated {len(prompts)} mixed prompts (seed={args.seed}).")

    targets = ["default", "cerebras"] if args.config == "both" else [args.config]
    summaries = []
    for cfg in targets:
        print(f"\nRunning '{cfg}' with {args.workers} workers...")
        summaries.append(_bench(cfg, prompts, args.workers))

    _print_report(summaries)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(summaries, f, indent=2)
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
