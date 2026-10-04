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
    asking_price
):
    # --- Input validation ---
    if asking_price <= 0:
        raise ValueError("asking_price must be greater than 0")
    if floor_area <= 0:
        raise ValueError("floor_area must be greater than 0")

    # --- Build input for the model (asking_price is NOT part of this) ---
    input_df = pd.DataFrame([{
        'Property Type': property_type,
        'County': county,
        'BER Rating': ber_rating,
        'Number of Bedrooms': bedrooms,
        'Number of Bathrooms': bathrooms,
        'Floor Area (m2)': floor_area,
        'Area_grouped': area_grouped
    }])

    # --- Get the model's predicted value ---
    log_prediction = model.predict(input_df)[0]
    predicted_value = np.expm1(log_prediction)

    # --- Compare asking price against predicted value ---
    difference = asking_price - predicted_value
    percentage_different = (difference / predicted_value) * 100

    overvalued = bool(percentage_different > 10)
    undervalued = bool(percentage_different < -10)  

    return {
        "predicted_value": round(float(predicted_value), 2),
        "asking_price": round(float(asking_price), 2),
        "percentage_difference": round(float(percentage_different), 2),
        "overvalued": overvalued,
        "undervalued": undervalued
    }