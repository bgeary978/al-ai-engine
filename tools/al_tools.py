"""CrewAI tool wrappers around the deterministic AL-AI Engine logic.

The LLM never calculates DTI/LTV/valuations itself. It calls these tools,
which run the existing, tested Python code, and then explains the output.
"""
import json
import sys
from pathlib import Path

import joblib
from crewai.tools import tool

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.affordability_agent import assess_affordability  # noqa: E402
from agents.valuation_agent import assess_valuation  # noqa: E402
from agents.market_researcher import research_market  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "models" / "rf_pipeline.joblib"

_model = None


def set_model(model):
    """Reuse the already-loaded Flask pipeline instead of loading a second copy."""
    global _model
    _model = model


def _get_model():
    """Return the shared valuation pipeline, loading from disk only if needed."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Valuation model not found at {MODEL_PATH}. "
                "Train/export rf_pipeline.joblib or set HF_MODEL_REPO for Flask."
            )
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
    first_time_buyer: bool = False,
) -> str:
    """Calculate monthly repayment, stress-tested repayment, DTI, LTV and LTI for a
    mortgage applicant and check them against Central Bank of Ireland limits.
    Returns JSON with repayments, ratios, deposit checks, net disposable surplus,
    within_regulatory_limits, regulatory_flags and warnings. Always use this tool
    for any affordability figures."""
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
    requested_loan_amount: float,
) -> str:
    """Compare a property's asking price with the machine-learning fair-value
    prediction and compute verified LTV against the AVM value. Returns JSON with
    predicted_value, asking_price, percentage_difference, overvalued, undervalued,
    verified_ltv and conservative_ltv. Always use this tool for any valuation
    figures."""
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
            requested_loan_amount=requested_loan_amount,
        )
        return json.dumps(result)
    except ValueError as e:
        return json.dumps({"error": str(e)})


@tool("Market Macro Researcher")
def market_research_tool(county: str = "Ireland") -> str:
    """Pull current ECB/base rates, typical Irish mortgage rates, house-price
    trend and market risk flags for a county using live web search (with a
    labelled static fallback if search fails). Returns JSON MacroContext.
    Always use this tool for market or macro figures."""
    ctx = research_market(county=county or "Ireland")
    payload = ctx.model_dump()
    payload.pop("snippets", None)
    payload["sources"] = (payload.get("sources") or [])[:5]
    return json.dumps(payload)
