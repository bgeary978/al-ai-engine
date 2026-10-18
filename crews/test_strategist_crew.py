"""Tests the Strategist (OpenAI) on three scenarios with known correct decisions.

Place this file at: al-ai-engine/crews/test_strategist_crew.py
Run from the project root:  python crews\\test_strategist_crew.py

The Affordability and Valuation inputs come straight from your tested Python
functions, so the only thing being tested here is the Strategist's decision.
"""
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.affordability_agent import assess_affordability  # noqa: E402
from agents.valuation_agent import assess_valuation  # noqa: E402
from tools.al_tools import _get_model  # noqa: E402
from crews.strategist_crew import build_strategist_crew  # noqa: E402

HOUSE = dict(
    property_type="House", county="Dublin", ber_rating="D1",
    bedrooms=4, bathrooms=4, floor_area=158, area_grouped="Other",
)

SCENARIOS = [
    {
        "name": "A: LTI breach, overvalued",
        "asking_price": 900000, "income": 85000, "debts": 300, "deposit": 90000,
    },
    {
        "name": "B: affordable but overvalued",
        "asking_price": 900000, "income": 200000, "debts": 0, "deposit": 180000,
    },
    {
        "name": "C: affordable and fairly priced",
        "asking_price": 690000, "income": 150000, "debts": 0, "deposit": 150000,
    },
]


def expected_decision(aff, val):
    """The lending policy, applied in plain Python (ground truth)."""
    if not aff["within_regulatory_limits"]:
        return "Decline"
    if val["overvalued"]:
        return "Conditionally Approve"
    return "Approve"


def parse_json(text):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    return json.loads(text)


passed = 0
for s in SCENARIOS:
    price = s["asking_price"]
    aff = assess_affordability(
        gross_annual_income=s["income"],
        existing_monthly_debts=s["debts"],
        deposit=s["deposit"],
        property_value=price,
        requested_loan_amount=price - s["deposit"],
        interest_rate=4.0,
        term_years=30,
        first_time_buyer=True,
    )
    val = assess_valuation(
        _get_model(), HOUSE["property_type"], HOUSE["county"], HOUSE["ber_rating"],
        HOUSE["bedrooms"], HOUSE["bathrooms"], HOUSE["floor_area"],
        HOUSE["area_grouped"], price,
    )
    want = expected_decision(aff, val)

    print(f"\n\n######## SCENARIO {s['name']} ########")
    print("Affordability:", aff)
    print("Valuation:", val)
    print("Expected decision:", want)

    raw = str(build_strategist_crew(aff, val).kickoff())
    try:
        out = parse_json(raw)
        got = out.get("decision")
    except Exception as e:  # noqa: BLE001
        print(f"\nCould not parse agent output as JSON: {e}\n{raw}")
        got = None

    ok = got == want
    passed += ok
    print(f"\n=== VERIFICATION {s['name']} ===")
    print(f"decision: {got!r} (expected {want!r}) -> {'OK' if ok else 'MISMATCH'}")

print(f"\n\n=== SUMMARY: {passed}/{len(SCENARIOS)} scenarios correct ===")