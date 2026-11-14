
import json
import os
import socket
import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from agents.affordability_agent import assess_affordability 
from agents.market_researcher import MacroContext  
from agents.valuation_agent import assess_valuation  
import app as app_module 

results = []
client = app_module.app.test_client()

BODY = {
    "gross_annual_income": 150000,
    "existing_monthly_debts": 0,
    "deposit": 150000,
    "property_value": 690000,
    "requested_loan_amount": 540000,
    "Property Type": "House",
    "County": "Cork",
    "BER Rating": "D1",
    "Number of Bedrooms": 4,
    "Number of Bathrooms": 4,
    "Floor Area (m2)": 158,
    "Area_grouped": "Other",
    "asking_price": 690000,
}

SCRIPTED_MACRO = MacroContext(
    ecb_main_refinancing_rate_pct=1.11,
    typical_variable_mortgage_rate_pct=8.88,
    typical_fixed_mortgage_rate_pct=7.77,
    house_price_trend="falling",
    risk_flags=["scripted rate check"],
    sources=["https://example.test/macro"],
    live_data=True,
)
AFFORDABILITY_TRAIL = "AFFORDABILITY TRAIL: calculator called"
VALUATION_TRAIL = "VALUATION TRAIL: model called"


def check(subtask, name, ok, detail):
    results.append((subtask, name, ok, detail))
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {subtask}: {name}")
    print(f"       {detail}")


def expected_decision(aff, val):
    if not aff["within_regulatory_limits"]:
        return "Decline"
    if val["overvalued"]:
        return "Conditionally Approve"
    return "Approve"


class FakeCrew:
    def __init__(self, text):
        self.text = text

    def kickoff(self):
        return self.text


def install_fakes(order, captured, strategist_text):
    def fake_research(county="Ireland"):
        order.append("research_market")
        captured["county"] = county
        return SCRIPTED_MACRO

    def fake_aff_crew(applicant):
        order.append("affordability_crew")
        captured["applicant_rate"] = applicant.get("interest_rate")
        captured["applicant_term"] = applicant.get("term_years")
        captured["applicant_ftb"] = applicant.get("first_time_buyer")
        return FakeCrew(AFFORDABILITY_TRAIL)

    def fake_val_crew(prop):
        order.append("valuation_crew")
        captured["prop_county"] = prop.get("county")
        return FakeCrew(VALUATION_TRAIL)

    def fake_strategist(affordability, valuation, macro_context=""):
        order.append("strategist_crew")
        captured["macro_context"] = macro_context
        captured["aff_limits"] = affordability.get("within_regulatory_limits")
        captured["overvalued"] = valuation.get("overvalued")
        return FakeCrew(strategist_text)

    researcher = types.ModuleType("agents.market_researcher")
    researcher.research_market = fake_research
    sys.modules["agents.market_researcher"] = researcher

    aff_mod = types.ModuleType("crews.affordability_crew")
    aff_mod.build_affordability_crew = fake_aff_crew
    sys.modules["crews.affordability_crew"] = aff_mod

    val_mod = types.ModuleType("crews.valuation_crew")
    val_mod.build_valuation_crew = fake_val_crew
    sys.modules["crews.valuation_crew"] = val_mod

    strat_mod = types.ModuleType("crews.strategist_crew")
    strat_mod.build_strategist_crew = fake_strategist
    sys.modules["crews.strategist_crew"] = strat_mod



missing_body = dict(BODY)
missing_body.pop("asking_price")
missing = client.post("/committee", json=missing_body)
missing_json = missing.get_json()
missing_ok = (
    missing.status_code == 400
    and isinstance(missing_json, dict)
    and "asking_price" in missing_json.get("error", "")
)
check(
    "Subtask 1",
    "Missing required field returns 400",
    missing_ok,
    f"status={missing.status_code} body={missing_json}",
)



invalid = dict(BODY)
invalid["gross_annual_income"] = 0
invalid_resp = client.post("/committee", json=invalid)
invalid_json = invalid_resp.get_json()
invalid_ok = invalid_resp.status_code == 400 and "gross_annual_income" in invalid_json.get("error", "")
check(
    "Subtask 2",
    "Invalid income returns 400 before any crew",
    invalid_ok,
    f"status={invalid_resp.status_code} body={invalid_json}",
)

direct_aff = assess_affordability(
    gross_annual_income=BODY["gross_annual_income"],
    existing_monthly_debts=BODY["existing_monthly_debts"],
    deposit=BODY["deposit"],
    property_value=BODY["property_value"],
    requested_loan_amount=BODY["requested_loan_amount"],
    interest_rate=4.0,
    term_years=30,
    first_time_buyer=True,
)
direct_val = assess_valuation(
    app_module.model,
    BODY["Property Type"],
    BODY["County"],
    BODY["BER Rating"],
    BODY["Number of Bedrooms"],
    BODY["Number of Bathrooms"],
    BODY["Floor Area (m2)"],
    BODY["Area_grouped"],
    BODY["asking_price"],
)
want = expected_decision(direct_aff, direct_val)
scripted_decision = {
    "decision": want,
    "reasons": [f"policy result {want}"],
    "conditions": ["independent valuation"] if want == "Conditionally Approve" else [],
    "credit_memo": f"Scripted memo for {want}.",
}

order = []
captured = {}
real_aff = app_module.assess_affordability
real_val = app_module.assess_valuation


def wrapped_aff(*args, **kwargs):
    order.append("assess_affordability")
    return real_aff(*args, **kwargs)


def wrapped_val(*args, **kwargs):
    order.append("assess_valuation")
    return real_val(*args, **kwargs)


app_module.assess_affordability = wrapped_aff
app_module.assess_valuation = wrapped_val
install_fakes(order, captured, json.dumps(scripted_decision))

success = client.post("/committee", json=BODY)
success_json = success.get_json()
macro_text = SCRIPTED_MACRO.to_prompt_text()
figures_match = (
    success.status_code == 200
    and success_json["affordability"] == direct_aff
    and success_json["valuation"] == direct_val
    and order[:2] == ["assess_affordability", "assess_valuation"]
    and captured.get("applicant_rate") == 4.0
    and captured.get("applicant_term") == 30
    and captured.get("applicant_ftb") is True
)
check(
    "Subtask 2",
    "Affordability and valuation run before the crews and match the calculators",
    figures_match,
    (
        f"order={order} status={success.status_code} "
        f"within_regulatory_limits={direct_aff['within_regulatory_limits']} flags={direct_aff['flags']} "
        f"predicted_value={direct_val['predicted_value']} percentage_difference={direct_val['percentage_difference']} "
        f"overvalued={direct_val['overvalued']} defaults rate={captured.get('applicant_rate')} "
        f"term={captured.get('applicant_term')} ftb={captured.get('applicant_ftb')}"
    ),
)



policy_detail = ""
policy_ok = False
try:
    real_mod = __import__("crews.strategist_crew", fromlist=["build_strategist_crew"])
    # The success request installed a fake module. Load the real file if needed.
    if not hasattr(real_mod, "LENDING_POLICY"):
        sys.modules.pop("crews.strategist_crew", None)
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "strategist_crew_real",
            PROJECT_ROOT / "crews" / "strategist_crew.py",
        )
        real_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(real_mod)
    crew = real_mod.build_strategist_crew(direct_aff, direct_val, macro_text)
    description = crew.tasks[0].description
    policy_ok = (
        "does not override" in description
        and "Decline" in description
        and "Conditionally Approve" in description
        and "Approve" in description
        and "1.11%" in description
        and "8.88%" in description
        and "7.77%" in description
    )
    policy_detail = (
        f"override_sentence={('does not override' in description)} "
        f"macro_in_prompt={'1.11%' in description} "
        f"county_passed={captured.get('county')} "
        f"strategist_received_macro={captured.get('macro_context') == macro_text}"
    )
except Exception as exc: 
    policy_detail = f"{type(exc).__name__}: {exc}"

macro_passed = (
    captured.get("county") == "Cork"
    and captured.get("macro_context") == macro_text
    and success.status_code == 200
    and success_json["macro"]["ecb_main_refinancing_rate_pct"] == 1.11
    and success_json["macro"]["typical_variable_mortgage_rate_pct"] == 8.88
    and success_json["macro"]["typical_fixed_mortgage_rate_pct"] == 7.77
    and success_json["macro"]["house_price_trend"] == "falling"
    and success_json["macro"]["live_data"] is True
)
check(
    "Subtask 3",
    "Market context is passed to the strategist and does not override policy",
    bool(policy_ok and macro_passed),
    policy_detail + f" response_macro_ecb={success_json.get('macro', {}).get('ecb_main_refinancing_rate_pct')} trend={success_json.get('macro', {}).get('house_price_trend')}",
)


trail_ok = (
    success.status_code == 200
    and success_json["decision"] == want
    and success_json["reasons"] == scripted_decision["reasons"]
    and success_json["conditions"] == scripted_decision["conditions"]
    and success_json["credit_memo"] == scripted_decision["credit_memo"]
    and success_json["affordability_trail"] == AFFORDABILITY_TRAIL
    and success_json["valuation_trail"] == VALUATION_TRAIL
)
check(
    "Subtask 4",
    "Success response contains the decision and the audit trail",
    trail_ok,
    (
        f"status={success.status_code} decision={success_json.get('decision')} "
        f"reasons={success_json.get('reasons')} conditions={success_json.get('conditions')} "
        f"credit_memo={success_json.get('credit_memo')!r} "
        f"affordability_trail={success_json.get('affordability_trail')!r} "
        f"valuation_trail={success_json.get('valuation_trail')!r}"
    ),
)

order_500 = []
captured_500 = {}

def raise_runtime(county="Ireland"):
    raise RuntimeError("LLM runtime missing")

sys.modules["agents.market_researcher"].research_market = raise_runtime
error_500 = client.post("/committee", json=BODY)
error_500_json = error_500.get_json()
runtime_ok = error_500.status_code == 500 and error_500_json.get("error") == "LLM runtime missing"
check(
    "Subtask 4",
    "Missing LLM runtime returns 500",
    runtime_ok,
    f"status={error_500.status_code} body={error_500_json}",
)

def fake_research_again(county="Ireland"):
    return SCRIPTED_MACRO

sys.modules["agents.market_researcher"].research_market = fake_research_again
sys.modules["crews.strategist_crew"].build_strategist_crew = (
    lambda affordability, valuation, macro_context="": FakeCrew("this is not json")
)
error_502 = client.post("/committee", json=BODY)
error_502_json = error_502.get_json()
bad_json_ok = (
    error_502.status_code == 502
    and error_502_json.get("error") == "Strategist did not return valid JSON"
    and error_502_json.get("affordability") == direct_aff
    and error_502_json.get("valuation") == direct_val
    and error_502_json.get("strategist_raw") == "this is not json"
)
check(
    "Subtask 4",
    "Invalid strategist JSON returns 502 with the assessments",
    bad_json_ok,
    (
        f"status={error_502.status_code} error={error_502_json.get('error')!r} "
        f"has_affordability={error_502_json.get('affordability') == direct_aff} "
        f"has_valuation={error_502_json.get('valuation') == direct_val} "
        f"strategist_raw={error_502_json.get('strategist_raw')!r}"
    ),
)



needed = [
    missing_ok,
    invalid_ok,
    figures_match,
    bool(policy_ok and macro_passed),
    trail_ok,
    runtime_ok,
    bad_json_ok,
]
check(
    "Subtask 5",
    "Verification test covers success, 400, 500, and 502",
    all(needed),
    f"{sum(needed)} of {len(needed)} checks passed. Success decision={success_json.get('decision')}.",
)

print("\n=== SUMMARY ===")
print(f"{sum(1 for _, _, ok, _ in results if ok)}/{len(results)} passed")

print("\n=== OPTIONAL LIVE COMMITTEE ===")
ollama_up = False
try:
    socket.create_connection(("127.0.0.1", 11434), 2).close()
    ollama_up = True
except OSError as exc:
    print(f"SKIPPED: Ollama is not reachable on port 11434 ({type(exc).__name__}).")

if ollama_up and not (os.getenv("GROQ_API_KEY") and os.getenv("OPENAI_API_KEY")):
    print("SKIPPED: GROQ_API_KEY or OPENAI_API_KEY is not set.")
elif ollama_up:
    print("SKIPPED: live POST /committee was not run in this pass. Mocked checks above do not call the LLMs.")

sys.exit(0 if all(ok for _, _, ok, _ in results) else 1)
