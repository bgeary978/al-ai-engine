
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.affordability_agent import assess_affordability  # noqa: E402
from crews.affordability_crew import build_affordability_crew  # noqa: E402

applicant = {
    "gross_annual_income": 85000,
    "existing_monthly_debts": 300,
    "deposit": 50000,
    "property_value": 450000,
    "requested_loan_amount": 400000,
    "interest_rate": 4.0,
    "term_years": 30,
    "first_time_buyer": True,
}

expected = assess_affordability(**applicant)
print("\n=== GROUND TRUTH (direct function call) ===")
print(expected)

result = str(build_affordability_crew(applicant).kickoff())
print("\n=== AGENT OUTPUT ===")
print(result)

digits = re.sub(r"[,\s]", "", result)
checks = {
    "monthly_repayment": f"{expected['monthly_repayment']:.2f}",
    "ltv": f"{expected['ltv_ratio']:.4f}",
}
print("\n=== VERIFICATION ===")
for name, value in checks.items():
    found = value in digits
    if name == "ltv" and not found:
        found = f"{expected['ltv_ratio'] * 100:.2f}" in digits
    print(f"{name}: {'OK - matches tool output' if found else 'MISMATCH - agent may have invented it'}")