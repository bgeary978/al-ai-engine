# agents/affordability_agent.py

def assess_affordability(
    gross_annual_income,
    existing_monthly_debts,
    deposit,
    property_value,
    requested_loan_amount,
    interest_rate=4.0,
    term_years=30,
    first_time_buyer=True
):
    # --- Input validation (add this block first) ---
    if gross_annual_income <= 0:
        raise ValueError("gross_annual_income must be greater than 0")
    if property_value <= 0:
        raise ValueError("property_value must be greater than 0")
    if requested_loan_amount <= 0:
        raise ValueError("requested_loan_amount must be greater than 0")
    if existing_monthly_debts < 0:
        raise ValueError("existing_monthly_debts cannot be negative")
    if deposit < 0:
        raise ValueError("deposit cannot be negative")
    if requested_loan_amount > property_value:
        raise ValueError("requested_loan_amount cannot exceed property_value")

    # --- Monthly mortgage repayment (standard amortization formula) ---
    monthly_rate = (interest_rate / 100) / 12
    num_payments = term_years * 12
    monthly_repayment = (
        requested_loan_amount * monthly_rate * (1 + monthly_rate) ** num_payments
    ) / ((1 + monthly_rate) ** num_payments - 1)

    # --- DTI: total monthly debt obligations / gross monthly income ---
    gross_monthly_income = gross_annual_income / 12
    total_monthly_debt = existing_monthly_debts + monthly_repayment
    dti_ratio = total_monthly_debt / gross_monthly_income

    # --- LTV: loan amount / property value ---
    ltv_ratio = requested_loan_amount / property_value

    # --- LTI: loan amount / gross annual income (Central Bank uses this too) ---
    lti_ratio = requested_loan_amount / gross_annual_income

    # --- Irish Central Bank macroprudential limits ---
    # Owner-occupier LTV cap is 90% for every buyer.
    # LTI is 4.0x for first-time buyers and 3.5x for others.
    max_ltv = 0.90
    max_lti = 4.0 if first_time_buyer else 3.5

    flags = []
    if ltv_ratio > max_ltv:
        flags.append(f"LTV ({ltv_ratio:.1%}) exceeds regulatory cap of {max_ltv:.0%}")
    if lti_ratio > max_lti:
        flags.append(f"LTI ({lti_ratio:.2f}x) exceeds regulatory cap of {max_lti}x income")
    if dti_ratio > 0.40:
        flags.append(f"DTI ({dti_ratio:.1%}) is high — above typical 40% affordability threshold")

    return {
        "monthly_repayment": round(monthly_repayment, 2),
        "dti_ratio": round(dti_ratio, 4),
        "ltv_ratio": round(ltv_ratio, 4),
        "lti_ratio": round(lti_ratio, 2),
        "within_regulatory_limits": len(flags) == 0,
        "flags": flags
    }