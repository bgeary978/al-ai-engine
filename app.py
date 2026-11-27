from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import re
import joblib
import pandas as pd
import numpy as np
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from agents.affordability_agent import assess_affordability
from agents.valuation_agent import assess_valuation

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

# Reject dossiers where stated price / financing figures drift too far apart.
PRICE_CONSISTENCY_TOLERANCE = 0.02  # 2%
TRAIL_MAX_CHARS = 2000

app = Flask(__name__)
CORS(app)

@app.get('/health')
def health():
    return 'ok'

def _load_valuation_model():
    """Prefer Hugging Face Hub when configured; otherwise use local joblib."""
    local_path = PROJECT_ROOT / "models" / "rf_pipeline.joblib"
    repo = os.environ.get("HF_MODEL_REPO")

    if repo:
        try:
            model_path = hf_hub_download(
                repo_id=repo,
                filename="rf_pipeline.joblib",
                token=os.environ.get("HF_TOKEN"),
            )
            return joblib.load(model_path)
        except Exception as exc:
            if not local_path.exists():
                raise RuntimeError(
                    f"Failed to download model from {repo} and no local "
                    f"fallback at {local_path}: {exc}"
                ) from exc

    if local_path.exists():
        return joblib.load(local_path)

    raise RuntimeError(
        "No valuation model available. Set HF_MODEL_REPO or place "
        f"rf_pipeline.joblib at {local_path}."
    )


model = _load_valuation_model()


def _rel_diff(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1.0)
    return abs(a - b) / denom


def _validate_price_consistency(property_value, asking_price, deposit, loan):
    """Ensure Mobile/Web don't feed an inconsistent loan dossier."""
    if _rel_diff(property_value, asking_price) > PRICE_CONSISTENCY_TOLERANCE:
        raise ValueError(
            f"property_value ({property_value}) and asking_price ({asking_price}) "
            f"differ by more than {PRICE_CONSISTENCY_TOLERANCE:.0%}"
        )
    financed = deposit + loan
    if _rel_diff(financed, property_value) > PRICE_CONSISTENCY_TOLERANCE:
        raise ValueError(
            f"deposit + requested_loan_amount ({financed}) must approximately "
            f"equal property_value ({property_value}) "
            f"(tolerance {PRICE_CONSISTENCY_TOLERANCE:.0%})"
        )


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    required_fields = [
        'Property Type', 'County', 'BER Rating', 'Number of Bedrooms',
        'Number of Bathrooms', 'Floor Area (m2)', 'Area_grouped',
    ]
    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({'error': f'Missing fields: {missing}'}), 400

    input_df = pd.DataFrame([{
        'Property Type': data['Property Type'],
        'County': data['County'],
        'BER Rating': data['BER Rating'],
        'Number of Bedrooms': data['Number of Bedrooms'],
        'Number of Bathrooms': data['Number of Bathrooms'],
        'Floor Area (m2)': data['Floor Area (m2)'],
        'Area_grouped': data['Area_grouped'],
    }])

    log_prediction = model.predict(input_df)[0]
    price_prediction = np.expm1(log_prediction)

    return jsonify({
        'predicted_price': round(float(price_prediction), 2)
    })


@app.route('/assess-affordability', methods=['POST'])
def affordability():
    data = request.get_json()

    required_fields = [
        'gross_annual_income', 'existing_monthly_debts', 'deposit',
        'property_value', 'requested_loan_amount',
    ]
    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({'error': f'Missing fields: {missing}'}), 400

    try:
        result = assess_affordability(
            gross_annual_income=data['gross_annual_income'],
            existing_monthly_debts=data['existing_monthly_debts'],
            deposit=data['deposit'],
            property_value=data['property_value'],
            requested_loan_amount=data['requested_loan_amount'],
            interest_rate=data.get('interest_rate', 4.0),
            term_years=data.get('term_years', 30),
            first_time_buyer=data.get('first_time_buyer', False),
        )
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    return jsonify(result)


@app.route('/assess-valuation', methods=['POST'])
def valuation():
    data = request.get_json()

    required_fields = [
        'Property Type', 'County', 'BER Rating', 'Number of Bedrooms',
        'Number of Bathrooms', 'Floor Area (m2)', 'Area_grouped', 'asking_price',
    ]
    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({'error': f'Missing fields: {missing}'}), 400

    try:
        result = assess_valuation(
            model=model,
            property_type=data['Property Type'],
            county=data['County'],
            ber_rating=data['BER Rating'],
            bedrooms=data['Number of Bedrooms'],
            bathrooms=data['Number of Bathrooms'],
            floor_area=data['Floor Area (m2)'],
            area_grouped=data['Area_grouped'],
            asking_price=data['asking_price'],
            requested_loan_amount=data.get('requested_loan_amount'),
        )
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    return jsonify(result)


def _parse_agent_json(text):
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    return json.loads(cleaned)


def _crew_output_text(result) -> str:
    """Use the task text only — not the full CrewAI object dump."""
    return str(getattr(result, "raw", None) or result).strip()


def _trim_trail(text, limit=TRAIL_MAX_CHARS):
    if not text:
        return text
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def _macro_public(macro):
    """Audit-sized market payload: drop search snippets, keep a few sources."""
    if macro is None:
        return None
    data = macro.model_dump()
    data.pop("snippets", None)
    data["sources"] = (data.get("sources") or [])[:5]
    return data


@app.route('/committee', methods=['POST'])
def committee():
    data = request.get_json(silent=True) or {}

    required_fields = [
        'gross_annual_income', 'existing_monthly_debts', 'deposit',
        'property_value', 'requested_loan_amount',
        'Property Type', 'County', 'BER Rating', 'Number of Bedrooms',
        'Number of Bathrooms', 'Floor Area (m2)', 'Area_grouped', 'asking_price',
    ]
    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({'error': f'Missing fields: {missing}'}), 400

    try:
        _validate_price_consistency(
            property_value=float(data['property_value']),
            asking_price=float(data['asking_price']),
            deposit=float(data['deposit']),
            loan=float(data['requested_loan_amount']),
        )
        affordability_result = assess_affordability(
            gross_annual_income=data['gross_annual_income'],
            existing_monthly_debts=data['existing_monthly_debts'],
            deposit=data['deposit'],
            property_value=data['property_value'],
            requested_loan_amount=data['requested_loan_amount'],
            interest_rate=data.get('interest_rate', 4.0),
            term_years=data.get('term_years', 30),
            first_time_buyer=data.get('first_time_buyer', False),
        )
        valuation_result = assess_valuation(
            model=model,
            property_type=data['Property Type'],
            county=data['County'],
            ber_rating=data['BER Rating'],
            bedrooms=data['Number of Bedrooms'],
            bathrooms=data['Number of Bathrooms'],
            floor_area=data['Floor Area (m2)'],
            area_grouped=data['Area_grouped'],
            asking_price=data['asking_price'],
            requested_loan_amount=data['requested_loan_amount'],
        )
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    applicant = {
        'gross_annual_income': data['gross_annual_income'],
        'existing_monthly_debts': data['existing_monthly_debts'],
        'deposit': data['deposit'],
        'property_value': data['property_value'],
        'requested_loan_amount': data['requested_loan_amount'],
        'interest_rate': data.get('interest_rate', 4.0),
        'term_years': data.get('term_years', 30),
        'first_time_buyer': data.get('first_time_buyer', False),
    }
    prop = {
        'property_type': data['Property Type'],
        'county': data['County'],
        'ber_rating': data['BER Rating'],
        'bedrooms': data['Number of Bedrooms'],
        'bathrooms': data['Number of Bathrooms'],
        'floor_area': data['Floor Area (m2)'],
        'area_grouped': data['Area_grouped'],
        'asking_price': data['asking_price'],
        'requested_loan_amount': data['requested_loan_amount'],
    }

    affordability_trail = None
    valuation_trail = None
    market_trail = None
    strategist_raw = None
    macro = None
    try:
        from agents.market_researcher import research_market
        from crews.affordability_crew import build_affordability_crew
        from crews.valuation_crew import build_valuation_crew
        from crews.market_crew import build_market_crew
        from crews.strategist_crew import build_strategist_crew
        from tools.al_tools import set_model

        set_model(model)
        macro = research_market(county=data['County'])
        affordability_trail = _trim_trail(
            _crew_output_text(build_affordability_crew(applicant).kickoff())
        )
        valuation_trail = _trim_trail(
            _crew_output_text(build_valuation_crew(prop).kickoff())
        )
        market_trail = _trim_trail(
            _crew_output_text(build_market_crew(data['County']).kickoff())
        )
        strategist_raw = _crew_output_text(build_strategist_crew(
            affordability_result,
            valuation_result,
            macro.to_prompt_text(),
        ).kickoff())
        decision = _parse_agent_json(strategist_raw)
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 500
    except json.JSONDecodeError:
        return jsonify({
            'error': 'Strategist did not return valid JSON',
            'affordability': affordability_result,
            'valuation': valuation_result,
            'macro': _macro_public(macro),
            'affordability_trail': affordability_trail,
            'valuation_trail': valuation_trail,
            'market_trail': market_trail,
            'strategist_raw': _trim_trail(strategist_raw),
        }), 502

    return jsonify({
        'affordability': affordability_result,
        'valuation': valuation_result,
        'macro': _macro_public(macro),
        'affordability_trail': affordability_trail,
        'valuation_trail': valuation_trail,
        'market_trail': market_trail,
        'decision': decision.get('decision'),
        'reasons': decision.get('reasons', []),
        'conditions': decision.get('conditions', []),
        'credit_memo': decision.get('credit_memo', ''),
    })


if __name__ == '__main__':
    app.run(debug=True)
