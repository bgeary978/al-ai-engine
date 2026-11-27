
import os
import re
import sys
from datetime import date
from pathlib import Path
from typing import Literal

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

SERPER_URL = "https://google.serper.dev/search"


class MacroContext(BaseModel):
    ecb_main_refinancing_rate_pct: float | None = None
    typical_variable_mortgage_rate_pct: float | None = None
    typical_fixed_mortgage_rate_pct: float | None = None
    house_price_trend: Literal["rising", "flat", "falling", "unknown"] = "unknown"
    risk_flags: list[str] = Field(default_factory=list)
    snippets: list[str] = Field(default_factory=list)  
    sources: list[str] = Field(default_factory=list)
    live_data: bool = True                              
    as_of: str = Field(default_factory=lambda: date.today().isoformat())

    def to_prompt_text(self) -> str:
        """Text block injected into the Underwriting Strategist's prompt."""
        def f(v):
            return f"{v}%" if v is not None else "unknown"
        lines = [
            f"MARKET & MACRO CONTEXT (as of {self.as_of}, "
            f"{'live web data' if self.live_data else 'STATIC FALLBACK - not live'}):",
            f"- ECB main refinancing rate: {f(self.ecb_main_refinancing_rate_pct)}",
            f"- Typical variable mortgage rate: {f(self.typical_variable_mortgage_rate_pct)}",
            f"- Typical fixed mortgage rate: {f(self.typical_fixed_mortgage_rate_pct)}",
            f"- House price trend: {self.house_price_trend}",
        ]
        if self.risk_flags:
            lines.append("- Risk flags: " + "; ".join(self.risk_flags))
        lines.append("- Sources: " + ", ".join(self.sources[:5]))
        return "\n".join(lines)



def _serper(query: str, num: int = 5) -> list[dict]:
    resp = requests.post(
        SERPER_URL,
        headers={"X-API-KEY": os.environ["SERPER_API_KEY"], "Content-Type": "application/json"},
        json={"q": query, "gl": "ie", "num": num},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    results = []
    if "answerBox" in data:  
        ab = data["answerBox"]
        results.append({"snippet": ab.get("answer") or ab.get("snippet", ""),
                        "link": ab.get("link", "")})
    results += [{"snippet": r.get("snippet", ""), "link": r.get("link", "")}
                for r in data.get("organic", [])]
    return results


def _first_rate(results: list[dict], must_contain: list[str]) -> float | None:
    """Percentage (0-15%) closest to a keyword in the first matching snippet."""
    for r in results:
        text = r["snippet"]
        lower = text.lower()
        hits = [lower.find(k) for k in must_contain if k in lower]
        if not hits:
            continue
        keyword_at = min(hits)
        best = None
        best_dist = None
        for m in re.finditer(r"(\d{1,2}(?:\.\d{1,2})?)\s?%", text):
            val = float(m.group(1))
            if not (0 < val < 15):
                continue
            dist = abs(m.start() - keyword_at)
            if best_dist is None or dist < best_dist:
                best = val
                best_dist = dist
        if best is not None:
            return best
    return None


def _trend(results: list[dict]) -> str:
    text = " ".join(r["snippet"].lower() for r in results)
    up = sum(text.count(w) for w in ("rise", "rose", "increase", "up ", "higher", "growth"))
    down = sum(text.count(w) for w in ("fall", "fell", "decrease", "down ", "lower", "drop"))
    if up == down == 0:
        return "unknown"
    if up > down * 1.5:
        return "rising"
    if down > up * 1.5:
        return "falling"
    return "flat"



def research_market(county: str = "Ireland") -> MacroContext:
    """Run the researcher. Never raises: falls back to a labelled static snapshot."""
    try:
        ecb = _serper("ECB main refinancing operations rate current")
        mort = _serper("average mortgage interest rates Ireland variable fixed this month")
        prices = _serper(f"house prices {county} Ireland latest year on year change")
        news = _serper("Ireland mortgage market risk central bank rate outlook news")

        ctx = MacroContext(
            ecb_main_refinancing_rate_pct=_first_rate(ecb, ["refinancing", "ecb", "main"]),
            typical_variable_mortgage_rate_pct=_first_rate(mort, ["variable"]),
            typical_fixed_mortgage_rate_pct=_first_rate(mort, ["fixed"]),
            house_price_trend=_trend(prices),
            snippets=[],
            sources=list(dict.fromkeys(
                r["link"] for r in (ecb + mort + prices + news) if r["link"]
            ))[:5],
        )
        text = " ".join(r["snippet"].lower() for r in news)
        for kw, flag in [("rate hike", "possible rate hikes"), ("downturn", "market downturn mentioned"),
                         ("recession", "recession risk mentioned"), ("arrears", "arrears pressure mentioned")]:
            if kw in text:
                ctx.risk_flags.append(flag)
        return ctx
    except Exception as exc:  
        return MacroContext(
            ecb_main_refinancing_rate_pct=2.65,      
            typical_variable_mortgage_rate_pct=4.30,  
            typical_fixed_mortgage_rate_pct=3.47,      
            house_price_trend="rising",
            risk_flags=[f"live research failed: {type(exc).__name__}"],
            live_data=False,
        )


if __name__ == "__main__":
    print(research_market("Cork").to_prompt_text())
