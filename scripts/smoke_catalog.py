#!/usr/bin/env python
"""Live verification of every catalog entry; the repo's record of what was checked.

For each entry, one run checks:

- call: one synchronous call through the library at the entry's own settings
  (which also proves the entry's effort setting is accepted);
- temperature: whether the model accepts temperature=0.0 when it is actually sent
  (raw samples at 0 and a high temperature are kept for reference), and whether
  identical requests come back as the same cached response;
- price: the entry's `pricing` against the provider's current rates (Vertex: Cloud
  Billing Catalog API, falling back to the public pricing page; Stanford: billed
  cost read back from the gateway's usage API).

Results are merged into src/inference/catalog/verification.json and rendered to
src/inference/catalog/VERIFIED.md. An entry whose call AND price check both pass
gets `verified: <today>` in its YAML (surfaced as ModelCard.verified_on); an entry
that fails either loses it.

Needs gcloud Application Default Credentials for Vertex entries and
STANFORD_AI_API_KEY for Stanford entries (skipped without it). Don't use the
Stanford key elsewhere while this runs: its price check reads key-wide usage.

    uv run scripts/smoke_catalog.py
    uv run scripts/smoke_catalog.py --only claude_opus_5 gemini_3_5_flash
    uv run scripts/smoke_catalog.py --no-write     # check only, change no files
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeout
from pathlib import Path
from typing import Any

from catalog_prices import VertexPrices, stanford_check

from inference import Prompt, llm
from inference.providers.claude_vertex import AnthropicVertex, ClaudeVertexModel

CATALOG = Path(__file__).resolve().parents[1] / "src" / "inference" / "catalog"
RECORD_JSON = CATALOG / "verification.json"
RECORD_MD = CATALOG / "VERIFIED.md"
PROMPT = Prompt(system_prompt="Be terse.", model_prompt="Reply with the word pong.")


def _error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {str(exc).replace(chr(10), ' ')[:300]}"


def _effort(model: Any) -> str | None:
    if not model.thinking:
        return None
    if isinstance(model, ClaudeVertexModel):
        if model.thinking_api == "adaptive":
            return f"adaptive/{model.effort}"
        return f"budget/{model.thinking_budget}"
    level = getattr(model, "thinking_level", None)
    if level is not None:
        return str(getattr(level, "value", level)).lower()
    return getattr(model, "reasoning_effort", None)


def _call_check(nickname: str) -> dict[str, Any]:
    start = time.monotonic()
    try:
        # A unique request id so an endpoint that caches responses has to answer live.
        prompt = PROMPT.model_copy(update={"system_prompt": f"Be terse. Request id: {uuid.uuid4()}"})
        out = llm.from_name(nickname).call(prompt)
    except Exception as exc:  # noqa: BLE001 - report every failure, keep going
        return {"ok": False, "error": _error(exc)}
    return {
        "ok": True,
        "seconds": round(time.monotonic() - start, 1),
        "text": out.text.strip()[:60],
        "thinking_text": bool(out.thinking),
        "tokens": [out.tokens.input_tokens, out.tokens.output_tokens,
                   out.tokens.thinking_tokens],
    }


def _temperature_check(nickname: str) -> str:
    """'accepted' or the rejection. ClaudeVertexModel only sends temperature with
    thinking off, so Claude is probed with the raw SDK, thinking off."""
    model = llm.from_name(nickname)
    try:
        if isinstance(model, ClaudeVertexModel):
            AnthropicVertex(project_id=model.project_id, region=model.location).messages.create(
                model=model.model_name,
                max_tokens=64,
                messages=[{"role": "user", "content": PROMPT.model_prompt}],
                temperature=0.0,
            )
        else:
            update = {"send_temperature": True} if hasattr(model, "send_temperature") else {}
            model.model_copy(update=update).call(PROMPT, temperature=0.0)
        return "accepted"
    except Exception as exc:  # noqa: BLE001
        return "rejected: " + _error(exc)


EFFECT_QUESTION = "Pick a random integer from 1 to 100. Reply with only the number."
EFFECT_SAMPLES = 5


def _effect_prompt() -> Prompt:
    # A unique request id makes every prompt distinct, so a gateway that caches
    # responses can't hand back the same stored reply. It is irrelevant to the
    # question, so at temperature 0 the answer should still barely move.
    return Prompt(system_prompt=f"Request id: {uuid.uuid4()}", model_prompt=EFFECT_QUESTION)


def _effect_model(nickname: str) -> tuple[Any, float]:
    """The entry configured so temperature is really sent (Claude with thinking off,
    since it's only sent then; OpenAI-compatible models with send_temperature on),
    plus the high temperature to compare against (Claude's maximum is 1.0)."""
    model = llm.from_name(nickname)
    update: dict[str, Any] = {}
    if hasattr(model, "send_temperature"):
        update["send_temperature"] = True
    if isinstance(model, ClaudeVertexModel):
        update["thinking"] = False
    high = 1.0 if isinstance(model, ClaudeVertexModel) else 1.5
    return model.model_copy(update=update), high


def _response_id(output: Any) -> str | None:
    payload = output.provider_response
    return getattr(payload, "id", None) or getattr(payload, "response_id", None)


def _cache_check(nickname: str) -> dict[str, Any]:
    """Send the identical request twice at a high temperature. Two different
    samples get two response ids; the same id twice means the endpoint returned a
    stored response instead of sampling again."""
    model, high = _effect_model(nickname)
    prompt = Prompt(system_prompt=None, model_prompt=EFFECT_QUESTION)
    try:
        first, second = (model.call(prompt, temperature=high) for _ in range(2))
        ids = [_response_id(first), _response_id(second)]
    except Exception as exc:  # noqa: BLE001
        return {"cached": None, "error": _error(exc)}
    if None in ids:
        return {"cached": None, "ids": ids}
    return {"cached": ids[0] == ids[1], "ids": ids}


def _temperature_samples(nickname: str) -> dict[str, Any]:
    """Raw answers to the same question (with a unique, irrelevant request id per
    call) at temperature 0 and at a high temperature, kept for reference only.
    They are not turned into an honored/ignored verdict: models collapse to favorite
    answers at any temperature, and some vary at 0 for other reasons, so the samples
    can't reliably show whether temperature is applied."""
    model, high = _effect_model(nickname)

    def answers(temperature: float) -> list[str]:
        return [
            " ".join(model.call(_effect_prompt(), temperature=temperature).text.split())
            for _ in range(EFFECT_SAMPLES)
        ]

    try:
        return {"at_0": answers(0.0), f"at_{high:g}": answers(high)}
    except Exception as exc:  # noqa: BLE001
        return {"error": _error(exc)}


def _live_checks(nickname: str) -> dict[str, Any]:
    model = llm.from_name(nickname)
    temperature = _temperature_check(nickname)
    return {
        "call": _call_check(nickname),
        "temperature": temperature,
        "temperature_samples": (
            _temperature_samples(nickname) if temperature == "accepted" else None
        ),
        "cache": _cache_check(nickname) if temperature == "accepted" else None,
        "effort": _effort(model),
    }


# --------------------------------------------------------------------------- #
# Record keeping
# --------------------------------------------------------------------------- #


def _set_verified(nickname: str, date: str | None) -> None:
    """Write or remove the `verified:` line in an entry's YAML, textually, so the
    rest of the file (key order, folded notes) is untouched."""
    path = CATALOG / f"{nickname}.yaml"
    lines = [ln for ln in path.read_text().splitlines() if not ln.startswith("verified:")]
    if date is not None:
        at = next((i for i, ln in enumerate(lines) if ln.startswith("notes:")), len(lines))
        lines.insert(at, f"verified: {date}")
    path.write_text("\n".join(lines) + "\n")


def _fmt_price(pair: Any) -> str:
    if not pair or None in pair:
        return "none"
    return f"${pair[0]:g} / ${pair[1]:g}"


def _render_md(record: dict[str, dict[str, Any]]) -> str:
    out = [
        "# Catalog verification",
        "",
        "Generated by `scripts/smoke_catalog.py`; do not edit by hand. An entry is",
        "**verified** on the last date its live call and its price check both passed",
        "(a price check that couldn't run keeps the previous date).",
        "temperature=0: whether the model accepts it (accepted doesn't prove it's applied).",
        "cached: whether two identical requests returned the same response id, i.e. the",
        "endpoint answers repeated requests from a cache instead of sampling again.",
        "Prices are USD per 1M input / output tokens.",
        "",
        "| entry | verified | call | temperature=0 | cached | effort | catalog price | found price | checked |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for nick in sorted(record):
        r = record[nick]
        call = r["call"]
        call_s = "ok" if call["ok"] else "FAIL"
        temp_s = "accepted" if r["temperature"] == "accepted" else "rejected"
        cache = (r.get("cache") or {}).get("cached")
        cache_s = {True: "yes", False: "no"}.get(cache, "—")
        price = r["price"]
        found = _fmt_price(price.get("found"))
        if price["ok"] is False:
            found += " ✗"
        elif price["ok"] is None:
            found = "unchecked"
        out.append(
            f"| {nick} | {r.get('verified') or '—'} | {call_s} | {temp_s} | {cache_s} | "
            f"{r['effort'] or '—'} | {_fmt_price(price.get('catalog'))} | {found} | "
            f"{r['checked']} |"
        )
    out += ["", "## Price evidence", ""]
    out += [f"- **{nick}**: {record[nick]['price']['source']}" for nick in sorted(record)]
    failures = [
        (nick, r) for nick, r in sorted(record.items())
        if not r["call"]["ok"] or r["price"]["ok"] is not True
    ]
    if failures:
        out += ["", "## Details for unverified entries", ""]
        for nick, r in failures:
            if not r["call"]["ok"]:
                out.append(f"- **{nick}** call: {r['call']['error']}")
            if r["price"]["ok"] is not True:
                out.append(f"- **{nick}** price: {r['price']['source']}")
    return "\n".join(out) + "\n"


def _print(nick: str, r: dict[str, Any]) -> None:
    call = r["call"]
    status = "OK" if call["ok"] else "FAIL"
    detail = (
        f"{call['seconds']:>5}s text={call['text']!r} tok={call['tokens']}"
        if call["ok"] else call["error"]
    )
    print(f"{status:<5} {nick:<24} {detail}")
    temp = r["temperature"] if r["temperature"] == "accepted" else r["temperature"][:120]
    samples = r.get("temperature_samples")
    if samples:
        temp += f" samples={samples}"
    if r.get("cache"):
        temp += f" cache={r['cache']}"
    print(f"      {'':<24} effort={r['effort']}  temperature: {temp}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="+", help="nicknames to check (default: all)")
    parser.add_argument("--no-write", action="store_true", help="change no files")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=300,
                        help="seconds before abandoning unfinished live checks")
    args = parser.parse_args()

    cards = {c.nickname: c for c in llm.cards()}
    nicknames = args.only or sorted(cards)
    unknown = [n for n in nicknames if n not in cards]
    if unknown:
        print(f"unknown nicknames: {unknown}")
        return 2
    stanford_key = os.environ.get("STANFORD_AI_API_KEY")
    skipped = [n for n in nicknames if cards[n].provider == "stanford" and not stanford_key]
    for n in skipped:
        print(f"SKIP  {n:<24} no STANFORD_AI_API_KEY")
    nicknames = [n for n in nicknames if n not in skipped]
    today = dt.date.today().isoformat()

    # Prices first: the Stanford check reads key-wide usage, so it must run before
    # the parallel live checks add their own usage to the same key.
    prices: dict[str, dict[str, Any]] = {}
    stanford = [n for n in nicknames if cards[n].provider == "stanford"]
    if stanford:
        print(f"checking Stanford prices ({len(stanford)} model(s), one at a time; "
              "each waits for the usage API)...", flush=True)
    for n in stanford:
        prices[n] = stanford_check(cards[n], stanford_key)
        print(f"PRICE {n:<24} {prices[n]['ok']}  {_fmt_price(prices[n]['found'])}"
              f"  ({prices[n]['source'][:100]})", flush=True)
    vertex = [n for n in nicknames if cards[n].provider != "stanford"]
    if vertex:
        project = getattr(llm.from_name(vertex[0]), "project_id", None)
        vertex_prices = VertexPrices(quota_project=project)
        print("checking Vertex prices...", flush=True)
        for n in vertex:
            prices[n] = vertex_prices.check(cards[n])
            print(f"PRICE {n:<24} {prices[n]['ok']}  {_fmt_price(prices[n]['found'])}"
                  f"  ({prices[n]['source'][:100]})", flush=True)

    print(f"running live checks on {len(nicknames)} entries...", flush=True)
    pool = ThreadPoolExecutor(max_workers=args.workers)
    futures = {pool.submit(_live_checks, n): n for n in nicknames}
    live: dict[str, dict[str, Any]] = {}
    try:
        for future in as_completed(futures, timeout=args.timeout):
            n = futures[future]
            live[n] = future.result()
            _print(n, live[n])
    except FuturesTimeout:
        for future, n in futures.items():
            if n not in live:
                live[n] = {
                    "call": {"ok": False, "error": f"timed out after {args.timeout:.0f}s"},
                    "temperature": "not checked", "temperature_samples": None,
                    "cache": None,
                    "effort": None,
                }
                _print(n, live[n])

    record = json.loads(RECORD_JSON.read_text()) if RECORD_JSON.exists() else {}
    record = {k: v for k, v in record.items() if k in cards}  # drop removed entries
    for n in nicknames:
        passed = live[n]["call"]["ok"] and prices[n]["ok"] is True
        previous = record.get(n, {}).get("verified")
        # A price check that couldn't run (e.g. the usage API returned a 502) says
        # nothing new about the price, so it keeps the last verified date; only a
        # failed call or a definite price mismatch clears it.
        undetermined = live[n]["call"]["ok"] and prices[n]["ok"] is None
        verified = today if passed else (previous if undetermined else None)
        record[n] = {
            **live[n],
            "price": prices[n],
            "checked": today,
            "verified": verified,
        }
        if not args.no_write:
            _set_verified(n, verified)
        if previous and not verified:
            print(f"NOTE  {n} was verified {previous}; that date is now cleared")
        elif undetermined:
            print(f"NOTE  {n}: price check couldn't run; kept verified date {previous}")

    if not args.no_write:
        RECORD_JSON.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        RECORD_MD.write_text(_render_md(record))
        print(f"\nwrote {RECORD_JSON.relative_to(CATALOG.parents[2])} and "
              f"{RECORD_MD.relative_to(CATALOG.parents[2])}")
    verified = sum(1 for n in nicknames if record[n]["verified"])
    print(f"{verified}/{len(nicknames)} checked entries verified", flush=True)
    # Hung requests can't be cancelled; exit without waiting on their threads.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
