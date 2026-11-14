
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pydantic import ValidationError 

import agents.market_researcher as mr  
from agents.market_researcher import (
    MacroContext,
    _first_rate,
    _serper,
    _trend,
    research_market,
)

results = []


def check(subtask, name, ok, detail):
    results.append((subtask, name, ok, detail))
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {subtask}: {name}")
    print(f"       {detail}")


def _fake_response(payload):
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = payload
    return response



real_serper_key = os.getenv("SERPER_API_KEY")
os.environ["SERPER_API_KEY"] = real_serper_key or "test-key-not-live"
captured = {}


def _capturing_post(url, headers=None, json=None, timeout=None):
    captured["url"] = url
    captured["headers"] = headers
    captured["json"] = json
    captured["timeout"] = timeout
    return _fake_response({
        "answerBox": {"answer": "The ECB main refinancing rate is 2.15%.", "link": "https://www.ecb.europa.eu/"},
        "organic": [{"snippet": "Variable mortgage rates averaged 4.20%.", "link": "https://www.centralbank.ie/"}],
    })


mr.requests.post = _capturing_post
rows = _serper("ECB main refinancing operations rate current", num=3)
key_sent = captured["headers"].get("X-API-KEY") == os.environ["SERPER_API_KEY"]
shape_ok = (
    captured["url"] == mr.SERPER_URL
    and captured["json"] == {"q": "ECB main refinancing operations rate current", "gl": "ie", "num": 3}
    and captured["timeout"] == 15
    and key_sent
    and len(rows) == 2
    and rows[0]["snippet"].startswith("The ECB main refinancing rate")
    and rows[0]["link"].startswith("https://www.ecb.europa.eu/")
    and "snippet" in rows[1]
    and "link" in rows[1]
)
check(
    "Subtask 1",
    "Serper search posts the query and reads the answer box plus organic results",
    shape_ok,
    f"{len(rows)} results. url={captured['url']} locale={captured['json'].get('gl')} timeout={captured['timeout']}",
)


schema_detail = ""
schema_ok = False
try:
    ctx = MacroContext(
        ecb_main_refinancing_rate_pct=2.15,
        typical_variable_mortgage_rate_pct=4.2,
        typical_fixed_mortgage_rate_pct=3.8,
        house_price_trend="rising",
        risk_flags=["possible rate hikes"],
        snippets=["ECB rate 2.15%"],
        sources=["https://example.com"],
    )
    text = ctx.to_prompt_text()
    required = [
        "ecb_main_refinancing_rate_pct",
        "typical_variable_mortgage_rate_pct",
        "typical_fixed_mortgage_rate_pct",
        "house_price_trend",
        "risk_flags",
        "snippets",
        "sources",
        "live_data",
        "as_of",
    ]
    missing = [name for name in required if name not in ctx.model_dump()]
    prompt_ok = all(piece in text for piece in ("2.15%", "4.2%", "3.8%", "rising", "possible rate hikes", "https://example.com", "live web data"))
    rejected = False
    try:
        MacroContext(house_price_trend="up")
    except ValidationError:
        rejected = True
    schema_ok = not missing and ctx.live_data is True and rejected and prompt_ok
    schema_detail = f"fields={list(ctx.model_dump())} live_data={ctx.live_data} bad trend rejected={rejected}"
except Exception as exc:  # noqa: BLE001
    schema_detail = f"{type(exc).__name__}: {exc}"
check("Subtask 2", "MacroContext schema and prompt text", schema_ok, schema_detail)



rate_ecb = _first_rate(
    [{"snippet": "The ECB main refinancing rate is 2.50% today."}],
    ["ecb", "refinancing"],
)
rate_variable = _first_rate(
    [{"snippet": "Typical variable mortgage rate is 4.60%."}],
    ["variable"],
)
shared_rates = [{"snippet": "Average variable rate 4.60% and average fixed rate 3.90%."}]
rate_variable_shared = _first_rate(shared_rates, ["variable"])
rate_fixed_shared = _first_rate(shared_rates, ["fixed"])
trend_up = _trend([{"snippet": "House prices rose and growth continued higher."}])
trend_down = _trend([{"snippet": "Prices fell and the market drop continued."}])
trend_unknown = _trend([{"snippet": "Prices were unchanged."}])

queries = []


def _scripted_post(url, headers=None, json=None, timeout=None):
    query = json["q"]
    queries.append(query)
    if query.startswith("ECB"):
        payload = {"organic": [{"snippet": "The ECB main refinancing rate is 2.50%.", "link": "https://www.ecb.europa.eu/"}]}
    elif query.startswith("average mortgage"):
        payload = {"organic": [{"snippet": "Average variable rate 4.60% and average fixed rate 3.90%.", "link": "https://www.centralbank.ie/"}]}
    elif query.startswith("house prices"):
        payload = {"organic": [{"snippet": "Cork house prices rose again this year.", "link": "https://www.cso.ie/"}]}
    else:
        payload = {"organic": [{"snippet": "Analysts warn of a rate hike and recession risk.", "link": "https://www.irishtimes.com/"}]}
    return _fake_response(payload)


mr.requests.post = _scripted_post
live = research_market("Cork")
extract_ok = (
    rate_ecb == 2.5
    and rate_variable == 4.6
    and rate_variable_shared == 4.6
    and rate_fixed_shared == 3.9
    and trend_up == "rising"
    and trend_down == "falling"
    and trend_unknown == "unknown"
    and live.live_data is True
    and live.ecb_main_refinancing_rate_pct == 2.5
    and live.typical_variable_mortgage_rate_pct == 4.6
    and live.typical_fixed_mortgage_rate_pct == 3.9
    and live.house_price_trend == "rising"
    and "possible rate hikes" in live.risk_flags
    and "recession risk mentioned" in live.risk_flags
    and any("Cork" in q for q in queries)
)
check(
    "Subtask 3",
    "Rates, trend, and risk flags extracted from search results",
    extract_ok,
    (
        f"unit rates ECB={rate_ecb} variable={rate_variable} shared variable={rate_variable_shared} shared fixed={rate_fixed_shared}; "
        f"trends={trend_up}, {trend_down}, {trend_unknown}; "
        f"extracted ECB={live.ecb_main_refinancing_rate_pct} variable={live.typical_variable_mortgage_rate_pct} "
        f"fixed={live.typical_fixed_mortgage_rate_pct} live_data={live.live_data} trend={live.house_price_trend} flags={live.risk_flags}"
    ),
)



def _boom(query, num=5):
    raise TimeoutError("simulated outage")


mr._serper = _boom
fallback = research_market("Cork")
prompt = fallback.to_prompt_text()
fallback_ok = (
    fallback.live_data is False
    and fallback.ecb_main_refinancing_rate_pct == 2.65
    and fallback.typical_variable_mortgage_rate_pct == 4.3
    and fallback.typical_fixed_mortgage_rate_pct == 3.47
    and fallback.house_price_trend == "rising"
    and any("TimeoutError" in flag for flag in fallback.risk_flags)
    and "STATIC FALLBACK - not live" in prompt
)
check(
    "Subtask 4",
    "Static fallback is returned instead of raising",
    fallback_ok,
    f"live_data={fallback.live_data} rates=({fallback.ecb_main_refinancing_rate_pct}, {fallback.typical_variable_mortgage_rate_pct}, {fallback.typical_fixed_mortgage_rate_pct}) trend={fallback.house_price_trend} flags={fallback.risk_flags}",
)



paths_ok = live.live_data is True and fallback.live_data is False
check(
    "Subtask 5",
    "Verification test covers a successful mocked search and a failed search",
    paths_ok,
    (
        f"Success path live_data={live.live_data}. "
        f"Failure path live_data={fallback.live_data}. "
        "Subtask 3 records whether the extracted figures themselves are correct."
    ),
)

print("\n=== SUMMARY ===")
print(f"{sum(1 for _, _, ok, _ in results if ok)}/{len(results)} passed")

from importlib import reload  

reload(mr)
live_key = bool(real_serper_key)
if live_key:
    os.environ["SERPER_API_KEY"] = real_serper_key
print("\n=== OPTIONAL LIVE SEARCH ===")
if not live_key:
    print("SKIPPED: SERPER_API_KEY is not set. Mocked checks above do not need it.")
    sys.exit(0 if all(ok for _, _, ok, _ in results) else 1)

try:
    sample = mr._serper("ECB main refinancing operations rate current", num=2)
    snippet = (sample[0].get("snippet") or "")[:180] if sample else "no results"
    print(f"LIVE PASS: {len(sample)} results. First snippet: {snippet}")
except Exception as exc:  # noqa: BLE001
    print(f"LIVE FAIL: {type(exc).__name__}: {exc}")
    sys.exit(1)

sys.exit(0 if all(ok for _, _, ok, _ in results) else 1)
