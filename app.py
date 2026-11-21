from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import re
import joblib
import pandas as pd
import numpy as np
from agents.affordability_agent import assess_affordability
from agents.valuation_agent import assess_valuation

app = Flask(__name__)
CORS(app)
@app.get('/health')
def health():
    return 'ok'
import os
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id=os.environ["HF_MODEL_REPO"],
    filename="rf_pipeline.joblib",
    token=os.environ.get("HF_TOKEN"),
)
model = joblib.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    required_fields = ['Property Type', 'County', 'BER Rating', 'Number of Bedrooms', 'Number of Bathrooms', 'Floor Area (m2)', 'Area_grouped']
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
        'Area_grouped': data['Area_grouped']
    }])

    log_prediction = model.predict(input_df)[0]
    price_prediction = np.expm1(log_prediction)

    return jsonify({
        'predicted_price': round(float(price_prediction), 2)
    })

@app.route('/assess-affordability', methods=['POST'])
def affordability():
    data = request.get_json()

    required_fields = ['gross_annual_income', 'existing_monthly_debts', 'deposit', 'property_value', 'requested_loan_amount']
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
            first_time_buyer=data.get('first_time_buyer', True)
        )
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    return jsonify(result)


@app.route('/assess-valuation', methods=['POST'])
def valuation():
    data = request.get_json()

    required_fields = ['Property Type', 'County', 'BER Rating', 'Number of Bedrooms',
                        'Number of Bathrooms', 'Floor Area (m2)', 'Area_grouped', 'asking_price']
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
            asking_price=data['asking_price']
        )
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    return jsonify(result)


def _parse_agent_json(text):
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    return json.loads(cleaned)


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
        affordability_result = assess_affordability(
            gross_annual_income=data['gross_annual_income'],
            existing_monthly_debts=data['existing_monthly_debts'],
            deposit=data['deposit'],
            property_value=data['property_value'],
            requested_loan_amount=data['requested_loan_amount'],
            interest_rate=data.get('interest_rate', 4.0),
            term_years=data.get('term_years', 30),
            first_time_buyer=data.get('first_time_buyer', True),
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
        'first_time_buyer': data.get('first_time_buyer', True),
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
    }

    affordability_trail = None
    valuation_trail = None
    strategist_raw = None
    macro = None
    try:
        from agents.market_researcher import research_market
        from crews.affordability_crew import build_affordability_crew
        from crews.valuation_crew import build_valuation_crew
        from crews.strategist_crew import build_strategist_crew

        macro = research_market(county=data['County'])
        affordability_trail = str(build_affordability_crew(applicant).kickoff())
        valuation_trail = str(build_valuation_crew(prop).kickoff())
        strategist_raw = str(build_strategist_crew(
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
            'macro': macro.model_dump() if macro else None,
            'affordability_trail': affordability_trail,
            'valuation_trail': valuation_trail,
            'strategist_raw': strategist_raw,
        }), 502

    return jsonify({
        'affordability': affordability_result,
        'valuation': valuation_result,
        'macro': macro.model_dump(),
        'affordability_trail': affordability_trail,
        'valuation_trail': valuation_trail,
        'decision': decision.get('decision'),
        'reasons': decision.get('reasons', []),
        'conditions': decision.get('conditions', []),
        'credit_memo': decision.get('credit_memo', ''),
    })


if __name__ == '__main__':
    app.run(debug=True)