import pandas as pd
import numpy as np


def assess_valuation(
    model,
    property_type,
    county,
    ber_rating,
    bedrooms,
    bathrooms,
    floor_area,
    area_grouped,
    asking_price,
    requested_loan_amount=None,
):
    if asking_price <= 0:
        raise ValueError("asking_price must be greater than 0")
    if floor_area <= 0:
        raise ValueError("floor_area must be greater than 0")
    if requested_loan_amount is not None and requested_loan_amount <= 0:
        raise ValueError("requested_loan_amount must be greater than 0")

    input_df = pd.DataFrame([{
        'Property Type': property_type,
        'County': county,
        'BER Rating': ber_rating,
        'Number of Bedrooms': bedrooms,
        'Number of Bathrooms': bathrooms,
        'Floor Area (m2)': floor_area,
        'Area_grouped': area_grouped
    }])

    log_prediction = model.predict(input_df)[0]
    predicted_value = float(np.expm1(log_prediction))

    difference = asking_price - predicted_value
    percentage_different = (difference / predicted_value) * 100

    overvalued = bool(percentage_different > 10)
    undervalued = bool(percentage_different < -10)

    # Conservative collateral: lesser of asking and AVM
    verified_collateral_value = min(float(asking_price), predicted_value)

    result = {
        "predicted_value": round(predicted_value, 2),
        "asking_price": round(float(asking_price), 2),
        "percentage_difference": round(float(percentage_different), 2),
        "overvalued": overvalued,
        "undervalued": undervalued,
        "verified_collateral_value": round(verified_collateral_value, 2),
    }

    if requested_loan_amount is not None:
        loan = float(requested_loan_amount)
        verified_ltv = loan / predicted_value
        conservative_ltv = loan / verified_collateral_value
        result["requested_loan_amount"] = round(loan, 2)
        result["verified_ltv"] = round(verified_ltv, 4)
        result["conservative_ltv"] = round(conservative_ltv, 4)
        # Policy should prefer the more conservative ratio
        result["verified_ltv_exceeds_90"] = conservative_ltv > 0.90
        result["conservative_ltv_exceeds_90"] = conservative_ltv > 0.90

    return result
