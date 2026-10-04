from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from agents.affordability_agent import assess_affordability
from agents.valuation_agent import assess_valuation

app = Flask(__name__)
CORS(app)
model = joblib.load('models/rf_pipeline.joblib')

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

if __name__ == '__main__':
    app.run(debug=True)