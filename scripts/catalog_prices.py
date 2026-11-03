"""Price checks for scripts/smoke_catalog.py.

Vertex (Claude, Gemini, GPT-OSS): list prices from the Cloud Billing Catalog API,
falling back to the public Vertex pricing page when no SKU matches. Stanford: the
gateway publishes no machine-readable rates, so two deliberately lopsided calls are
made per model and their billed cost is read back from the usage API, then solved
for the input and output rate.

Every check returns a dict with ``ok`` (True / False / None for "couldn't check"),
the catalog rates, the found rates, and where they came from.
"""

from __future__ import annotations

import datetime as dt
import random
import re
import time
import uuid
from html.parser import HTMLParser
from typing import Any

import google.auth
import httpx
from google.auth.transport.requests import AuthorizedSession

from inference import LLMOutput, ModelCard, Prompt, llm

TOLERANCE = 0.01  # relative; rates within 1% of the catalog count as a match

# --------------------------------------------------------------------------- #
# Vertex
# --------------------------------------------------------------------------- #

_BILLING = "https://cloudbilling.googleapis.com/v1"
# Gemini models are on the first page; Claude and GPT-OSS on the partner page. Both
# list the standard tier first, then priority and flex/batch tabs.
_PRICING_PAGES = (
    "https://cloud.google.com/vertex-ai/generative-ai/pricing",
    "https://cloud.google.com/model-garden/partner-models/pricing",
)
# Services whose SKUs can carry Vertex model token prices. Partner models (Claude,
# GPT-OSS) may be billed under their own service rather than "Vertex AI".
_SERVICE_PATTERN = re.compile(r"vertex|gemini|claude|anthropic|gpt.?oss", re.I)
# SKU description words that mean "not the standard text-token rate".
_EXCLUDE = re.compile(
    r"batch|cache|cached|caching|long|>|200k|272k|audio|image|video|tuning|"
    r"grounding|provisioned|priority|flex|storage|regional",
    re.I,
)
# Only rows under the provider's own model-family section count; other sections
# (CodeMender, AlphaEvolve, tuning, caching storage, ...) reuse model names with
# different prices.
_PAGE_SECTIONS = {
    "claude_vertex": re.compile(r"claude models", re.I),
    "gemini_vertex": re.compile(r"^gemini \d+$", re.I),
    "gpt_oss_vertex": re.compile(r"openai.s models", re.I),
}
_INPUT_TYPE = re.compile(r"^input(?: tokens| \((?=[^)]*text)[^)]*\))?$")
_OUTPUT_TYPE = re.compile(r"^(?:text )?output(?: tokens| \([^)]*\))?$")
# Promotional qualifiers glued onto the model cell, e.g. "Gemini 3.8 Flash*through
# December 31, 2026" (current price) vs "...Starting January 1, 2027" (future price).
_QUALIFIER = re.compile(r"\s*\*?\s*((?:through|starting) [a-z]+ \d+, \d{4})$", re.I)


class _TableRows(HTMLParser):
    """Every <tr> on a page as (section heading, [cell text, ...])."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[tuple[str, list[str]]] = []
        self._heading: str = ""
        self._in_heading: str | None = None
        self._heading_text: list[str] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in ("h2", "h3", "h4"):
            self._in_heading, self._heading_text = tag, []
        elif tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag == self._in_heading:
            self._heading = " ".join("".join(self._heading_text).split())
            self._in_heading = None
        elif tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append((self._heading, self._row))
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._in_heading:
            self._heading_text.append(data)
        if self._cell is not None:
            self._cell.append(data)


def _norm(text: str) -> str:
    return re.sub(r"[\s_\-]+", " ", text.lower()).strip()


def _model_phrase(card: ModelCard) -> str:
    """'claude opus 4.8' / 'gemini 3 flash' / 'gpt oss 120b' -- the card name with any
    parenthetical dropped."""
    return _norm(re.sub(r"\(.*?\)", "", card.name))


def _phrase_regex(phrase: str) -> re.Pattern[str]:
    # Not preceded/followed by a word char or '.', and not followed by a variant
    # word, so "gemini 3 flash" matches neither "gemini 3.1 flash" nor "gemini 3
    # flash lite" / "... image" / "... live api".
    # Newer partner-page rows drop the vendor ("Opus 5", not "Claude Opus 5").
    body = re.escape(phrase)
    if phrase.startswith("claude "):
        body = r"(?:claude )?" + re.escape(phrase.removeprefix("claude "))
    return re.compile(r"(?<![\w.])" + body + r"(?![\w.]| lite| image| live| cyber)")


class VertexPrices:
    """Lazily loads Vertex SKUs once per run (and the pricing pages if needed)."""

    def __init__(self, quota_project: str | None) -> None:
        self._quota_project = quota_project
        self._skus: list[dict[str, Any]] | None = None
        self._skus_error: str | None = None
        self._page_rows: list[tuple[str, str, list[str]]] | None = None

    def _session(self) -> AuthorizedSession:
        creds, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        if self._quota_project and hasattr(creds, "with_quota_project"):
            creds = creds.with_quota_project(self._quota_project)
        return AuthorizedSession(creds)

    def _paged(self, session: AuthorizedSession, url: str, key: str) -> list[dict]:
        items: list[dict] = []
        params: dict[str, str] = {"pageSize": "5000"}
        while True:
            response = session.get(url, params=params, timeout=60)
            if response.status_code == 403:
                message = response.json().get("error", {}).get("message", "")
                raise PermissionError(message[:200])
            response.raise_for_status()
            body = response.json()
            items += body.get(key) or []
            token = body.get("nextPageToken")
            if not token:
                return items
            params["pageToken"] = token

    def skus(self) -> list[dict[str, Any]]:
        if self._skus is None and self._skus_error is None:
            try:
                session = self._session()
                services = [
                    s
                    for s in self._paged(session, f"{_BILLING}/services", "services")
                    if _SERVICE_PATTERN.search(s.get("displayName", ""))
                ]
                skus: list[dict[str, Any]] = []
                for service in services:
                    skus += self._paged(
                        session,
                        f"{_BILLING}/{service['name']}/skus?currencyCode=USD",
                        "skus",
                    )
                self._skus = skus
            except Exception as exc:  # noqa: BLE001 - fall back to the pages
                self._skus_error = f"{type(exc).__name__}: {exc}"[:300]
                print(f"billing catalog unavailable, using pricing pages: {self._skus_error}")
        return self._skus or []

    def page_rows(self) -> list[tuple[str, str, list[str]]]:
        """(page url, section heading, cells) for every table row on both pages."""
        if self._page_rows is None:
            rows = []
            for url in _PRICING_PAGES:
                parser = _TableRows()
                parser.feed(httpx.get(url, timeout=60, follow_redirects=True).text)
                rows += [(url, heading, cells) for heading, cells in parser.rows]
            self._page_rows = rows
        return self._page_rows

    # -- lookups ------------------------------------------------------------- #

    def _sku_rate(self, card: ModelCard, direction: str) -> tuple[float, str] | str:
        """(USD per 1M tokens, SKU description) or an explanation of why not."""
        pattern = _phrase_regex(_model_phrase(card))
        candidates = []
        for sku in self.skus():
            desc = _norm(sku.get("description", ""))
            if not pattern.search(desc) or direction not in desc or _EXCLUDE.search(desc):
                continue
            regions = sku.get("serviceRegions") or []
            if regions and "global" not in regions:
                continue
            candidates.append(sku)
        if not candidates:
            return "no matching SKU"
        if len(candidates) > 1:
            names = "; ".join(sorted({c["description"] for c in candidates}))
            return f"ambiguous SKUs: {names}"[:400]
        sku = candidates[0]
        expr = sku["pricingInfo"][0]["pricingExpression"]
        rate = next(r for r in expr["tieredRates"] if not r.get("startUsageAmount"))
        unit_price = float(rate["unitPrice"].get("units") or 0) + (
            rate["unitPrice"].get("nanos") or 0
        ) / 1e9
        factor = float(expr.get("baseUnitConversionFactor") or 1)
        per_million = unit_price / factor * 1e6
        return per_million, f"{sku['description']} [{expr.get('usageUnitDescription')}]"

    def _page_rates(self, card: ModelCard) -> tuple[float, float, str] | str:
        """(input, output, evidence) from the model's first price block in its
        provider's section, or why not. A block is the row naming the model plus the
        rows under it. The first block is the standard tier (priority and flex/batch
        tabs follow), or for a promotional model the "through <date>" block; blocks
        "starting <date>" are future prices and skipped. A first block missing its
        input or output row is a failure, never a fall-through to another tier."""
        section = _PAGE_SECTIONS.get(card.provider)
        if section is None:
            return f"no pricing-page section known for provider {card.provider}"
        name = _phrase_regex(_model_phrase(card))
        rows = self.page_rows()
        for i, (url, heading, cells) in enumerate(rows):
            if not cells or not cells[0] or not section.search(heading):
                continue
            qualifier = _QUALIFIER.search(cells[0])
            label = _QUALIFIER.sub("", cells[0])
            label = _norm(label.replace("*", "")).removesuffix(" preview").strip()
            if not name.fullmatch(label):
                continue
            if qualifier and qualifier.group(1).lower().startswith("starting"):
                continue
            block = [cells]
            for _, _, more in rows[i + 1 :]:
                if not more or more[0]:
                    break
                block.append(more)
            prices: dict[str, float] = {}
            kind = ""
            for row in block:
                kind = _norm(row[1]) if len(row) > 1 and row[1] else kind
                region = _norm(row[2]) if len(row) > 2 and not row[2].startswith("$") else ""
                if region and not region.startswith("global"):
                    continue
                price = next((c for c in row[1:] if c.startswith("$")), None)
                if price is None:
                    continue
                for key, pattern in (("input", _INPUT_TYPE), ("output", _OUTPUT_TYPE)):
                    if key not in prices and pattern.match(kind):
                        prices[key] = float(price.lstrip("$").replace(",", ""))
            if "input" not in prices or "output" not in prices:
                missing = " and ".join(k for k in ("input", "output") if k not in prices)
                return f"{url} [{heading}] lists '{cells[0]}' with no {missing} price row"
            evidence = (
                f"{url} [{heading}] row '{cells[0]}': "
                f"input ${prices['input']:g}, output ${prices['output']:g}"
            )
            return prices["input"], prices["output"], evidence
        return f"not listed in the {section.pattern!r} section of the Vertex pricing pages"

    def check(self, card: ModelCard) -> dict[str, Any]:
        catalog = (card.input_price_per_million, card.output_price_per_million)
        found_in = self._sku_rate(card, "input")
        found_out = self._sku_rate(card, "output")
        if isinstance(found_in, tuple) and isinstance(found_out, tuple):
            found = (found_in[0], found_out[0])
            source = f"billing catalog: {found_in[1]} | {found_out[1]}"
        else:
            try:
                page = self._page_rates(card)
            except Exception as exc:  # noqa: BLE001
                page = f"pricing page fetch failed: {type(exc).__name__}"
            if isinstance(page, str):
                return {"ok": None, "catalog": catalog, "found": None, "source": page}
            found, source = page[:2], f"pricing page: {page[2]}"
        return {"ok": _matches(catalog, found), "catalog": catalog, "found": found,
                "source": source}


# --------------------------------------------------------------------------- #
# Stanford
# --------------------------------------------------------------------------- #

_USAGE_URL = "https://aiapigateway-usage.stanford.edu/usage"
# Size each model's two calls so that even at a ceiling rate (4x catalog, or
# $5/$20 per 1M when unpriced) they cost at most this much.
_MODEL_BUDGET = 0.25
_A_SHARE = 0.6
_UNPRICED_CEILING = (5.00, 20.00)
_A_MAX_OUTPUT = 200
_B_MAX_OUTPUT = 8_000
_PROMPT_OVERHEAD = 200
_WORST_TOKENS_PER_WORD = 2
_MAX_FILLER_WORDS = 40_000  # stays far below the 272K long-context tier
_POLL_SECONDS = 20
_POLL_TIMEOUT = 15 * 60
_VOCAB = (
    "the of and to in is was for on are with as his they at be this from have or by "
    "one had not but what all were when we there can an your which their said if do "
    "will each about how up out them then she many some so these would other into has"
).split()


_USAGE_ATTEMPTS = 5


def _usage(api_key: str) -> tuple[float, int, int]:
    today = dt.date.today()
    # The usage API intermittently returns 5xx. It's a read-only lookup, so retry
    # with a growing pause rather than abandoning the whole price check.
    for attempt in range(1, _USAGE_ATTEMPTS + 1):
        response = httpx.get(
            _USAGE_URL,
            params={
                "start_date": (today - dt.timedelta(days=1)).isoformat(),
                "end_date": today.isoformat(),
            },
            headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
            timeout=30,
        )
        if response.status_code < 500 or attempt == _USAGE_ATTEMPTS:
            break
        time.sleep(10 * attempt)
    response.raise_for_status()
    rows = response.json().get("data") or []
    return (
        sum(r.get("total_cost") or 0 for r in rows),
        sum(r.get("total_input_tokens") or 0 for r in rows),
        sum(r.get("total_output_tokens") or 0 for r in rows),
    )


def _billed_cost(api_key: str, before: tuple[float, int, int], out: LLMOutput) -> float:
    """Poll until key-wide usage grows by exactly this call's tokens; return its cost."""
    want = (out.tokens.input_tokens, out.tokens.output_tokens)
    deadline = time.monotonic() + _POLL_TIMEOUT
    while True:
        now = _usage(api_key)
        delta = (now[1] - before[1], now[2] - before[2])
        if delta == want:
            return now[0] - before[0]
        if delta[0] > want[0] or delta[1] > want[1]:
            raise RuntimeError(
                f"usage grew by {delta}, more than this call's {want}; "
                "the key was used elsewhere during the check"
            )
        if time.monotonic() > deadline:
            raise RuntimeError(f"usage did not catch up (seen {delta}, want {want})")
        time.sleep(_POLL_SECONDS)


def stanford_check(card: ModelCard, api_key: str) -> dict[str, Any]:
    catalog = (card.input_price_per_million, card.output_price_per_million)
    try:
        return _stanford_measure(card, api_key, catalog)
    except Exception as exc:  # noqa: BLE001 - one model's failure must not end the run
        return {"ok": None, "catalog": catalog, "found": None,
                "source": f"usage-API probe failed: {type(exc).__name__}: {exc}"[:300]}


def _stanford_measure(
    card: ModelCard, api_key: str, catalog: tuple[float | None, float | None]
) -> dict[str, Any]:
    # Unpriced or $0 entries can't size the calls, so assume a generous ceiling.
    if None in catalog or 0 in catalog:
        cin, cout = _UNPRICED_CEILING
    else:
        cin, cout = catalog[0] * 4, catalog[1] * 4
    cin, cout = cin / 1e6, cout / 1e6
    words = int(
        (_MODEL_BUDGET * _A_SHARE - _PROMPT_OVERHEAD * cin - _A_MAX_OUTPUT * cout)
        / (_WORST_TOKENS_PER_WORD * cin)
    )
    words = min(words, _MAX_FILLER_WORDS)
    b_output = min(
        int((_MODEL_BUDGET * (1 - _A_SHARE) - _PROMPT_OVERHEAD * cin) / cout),
        _B_MAX_OUTPUT,
    )

    model = llm.from_name(card.nickname)
    # Low effort keeps reasoning from eating call A's small output cap; reasoning
    # bills as output either way, so it doesn't disturb the measurement.
    if getattr(model, "reasoning_effort", None) is not None:
        model = model.model_copy(update={"reasoning_effort": "low"})
    filler = " ".join(random.choice(_VOCAB) for _ in range(words))
    # A unique nonce up front defeats prompt caching, which bills input cheaper.
    prompt_a = Prompt(
        system_prompt="Be terse.",
        model_prompt=f"Request {uuid.uuid4()}. Ignore the filler below and reply "
        f"with only the word: done\n\n{filler}",
    )
    prompt_b = Prompt(
        system_prompt=None,
        model_prompt=f"Request {uuid.uuid4()}. Write the integers from 1 to 4000 in "
        "order, one per line, with no other text.",
    )
    before = _usage(api_key)
    out_a = model.call(prompt_a, max_tokens=_A_MAX_OUTPUT)
    cost_a = _billed_cost(api_key, before, out_a)
    before = _usage(api_key)
    out_b = model.call(prompt_b, max_tokens=b_output)
    cost_b = _billed_cost(api_key, before, out_b)

    ia, oa = out_a.tokens.input_tokens, out_a.tokens.output_tokens
    ib, ob = out_b.tokens.input_tokens, out_b.tokens.output_tokens
    det = ia * ob - ib * oa
    found = (  # "+ 0.0" turns a -0.0 from the solve into 0.0
        round((cost_a * ob - cost_b * oa) / det * 1e6, 4) + 0.0,
        round((ia * cost_b - ib * cost_a) / det * 1e6, 4) + 0.0,
    )
    return {"ok": _matches(catalog, found), "catalog": catalog, "found": found,
            "source": f"usage API, measured (spent ${cost_a + cost_b:.4f})"}


def _matches(catalog: tuple[float | None, float | None], found: tuple[float, float]) -> bool:
    if None in catalog:
        return False
    return all(abs(c - f) <= max(TOLERANCE * c, 1e-6) for c, f in zip(catalog, found))
