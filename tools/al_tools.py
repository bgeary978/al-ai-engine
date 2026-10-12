"""CrewAI tool wrappers around the deterministic AL-AI Engine logic.

Place this file at: al-ai-engine/tools/al_tools.py
(create tools/__init__.py as an empty file too)

The LLM never calculates DTI/LTV/valuations itself. It calls these tools,
which run the existing, tested Python code, and then explains the output.
"""
import json
import sys
from pathlib import Path

import joblib
from crewai.tools import tool

# Make the project root importable so `agents.*` resolves.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.affordability_agent import assess_affordability  # noqa: E402
from agents.valuation_agent import assess_valuation  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "models" / "rf_pipeline.joblib"

_model = None


def _get_model():
    """Load the trained valuation pipeline once and reuse it."""
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


@tool("Affordability Calculator")
def affordability_tool(
    gross_annual_income: float,
    existing_monthly_debts: float,
    deposit: float,
    property_value: float,
    requested_loan_amount: float,
    interest_rate: float = 4.0,
    term_years: int = 30,
    first_time_buyer: bool = True,
) -> str:
    """Calculate monthly repayment, DTI, LTV and LTI for a mortgage applicant and
    check them against Central Bank of Ireland limits. Returns JSON with
    monthly_repayment, dti_ratio, ltv_ratio, lti_ratio, within_regulatory_limits
    and a list of flags. Always use this tool for any affordability figures."""
    try:
        result = assess_affordability(
            gross_annual_income=gross_annual_income,
            existing_monthly_debts=existing_monthly_debts,
            deposit=deposit,
            property_value=property_value,
            requested_loan_amount=requested_loan_amount,
            interest_rate=interest_rate,
            term_years=term_years,
            first_time_buyer=first_time_buyer,
        )
        return json.dumps(result)
    except ValueError as e:
        return json.dumps({"error": str(e)})


@tool("Property Valuation Checker")
def valuation_tool(
    property_type: str,
    county: str,
    ber_rating: str,
    bedrooms: int,
    bathrooms: int,
    floor_area: float,
    area_grouped: str,
    asking_price: float,
) -> str:
    """Compare a property's asking price with the machine-learning fair-value
    prediction. Returns JSON with predicted_value, asking_price,
    percentage_difference, overvalued and undervalued (flagged at +/-10%).
    Always use this tool for any valuation figures."""
    try:
        result = assess_valuation(
            _get_model(),
            property_type,
            county,
            ber_rating,
            bedrooms,
            bathrooms,
            floor_area,
            area_grouped,
            asking_price,
        )
        return json.dumps(result)
    except ValueError as e:
        return json.dumps({"error": str(e)})
