# agents/affordability_agent.py

STRESS_BPS = 200  # +2.0 percentage points (Central Bank-style stress)

# Simple effective-tax haircut so "net disposable surplus" is not pure gross.
# Bands are illustrative Irish-style averages for a prototype (not a full tax engine).
EFFECTIVE_TAX_RATE_LOW = 0.20   # under ~€40k gross
EFFECTIVE_TAX_RATE_MID = 0.28  # €40k–€70k
EFFECTIVE_TAX_RATE_HIGH = 0.35  # above ~€70k


def _monthly_repayment(principal: float, annual_rate_pct: float, term_years: int) -> float:
    monthly_rate = (annual_rate_pct / 100) / 12
    num_payments = term_years * 12
    if monthly_rate == 0:
        return principal / num_payments
    return (
        principal * monthly_rate * (1 + monthly_rate) ** num_payments
    ) / ((1 + monthly_rate) ** num_payments - 1)


def _effective_tax_rate(gross_annual_income: float) -> float:
    if gross_annual_income < 40000:
        return EFFECTIVE_TAX_RATE_LOW
    if gross_annual_income < 70000:
        return EFFECTIVE_TAX_RATE_MID
    return EFFECTIVE_TAX_RATE_HIGH


def _estimated_net_monthly_income(gross_annual_income: float) -> float:
    rate = _effective_tax_rate(gross_annual_income)
    return (gross_annual_income * (1 - rate)) / 12


def assess_affordability(
    gross_annual_income,
    existing_monthly_debts,
    deposit,
    property_value,
    requested_loan_amount,
    interest_rate=4.0,
    term_years=30,
    first_time_buyer=False,
    stress_bps=STRESS_BPS,
):
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
    if interest_rate < 0:
        raise ValueError("interest_rate cannot be negative")
    if term_years is None or int(term_years) <= 0:
        raise ValueError("term_years must be greater than 0")
    if stress_bps < 0:
        raise ValueError("stress_bps cannot be negative")

    term_years = int(term_years)
    first_time_buyer = bool(first_time_buyer)

    monthly_repayment = _monthly_repayment(
        requested_loan_amount, interest_rate, term_years
    )

    gross_monthly_income = gross_annual_income / 12
    net_monthly_income = round(_estimated_net_monthly_income(gross_annual_income), 2)
    effective_tax_rate = _effective_tax_rate(gross_annual_income)

    total_monthly_debt = existing_monthly_debts + monthly_repayment
    # DTI still uses gross (common underwriting convention)
    dti_ratio = total_monthly_debt / gross_monthly_income

    # Net disposable surplus after estimated tax, existing debts, and mortgage
    net_disposable_surplus = round(net_monthly_income - total_monthly_debt, 2)

    ltv_ratio = requested_loan_amount / property_value
    lti_ratio = requested_loan_amount / gross_annual_income

    # CBI-style PDH caps: FTB 90% LTV / 4.0x LTI; non-FTB 80% LTV / 3.5x LTI
    max_ltv = 0.90 if first_time_buyer else 0.80
    max_lti = 4.0 if first_time_buyer else 3.5

    required_deposit = property_value - requested_loan_amount
    min_deposit_for_ltv_cap = property_value * (1 - max_ltv)

    stressed_rate = interest_rate + (stress_bps / 100.0)
    stressed_monthly_repayment = _monthly_repayment(
        requested_loan_amount, stressed_rate, term_years
    )
    stressed_total_debt = existing_monthly_debts + stressed_monthly_repayment
    stressed_dti_ratio = stressed_total_debt / gross_monthly_income
    stressed_net_disposable_surplus = round(
        net_monthly_income - stressed_total_debt, 2
    )

    regulatory_flags = []
    warnings = []

    if ltv_ratio > max_ltv:
        buyer = "first-time buyer" if first_time_buyer else "non-first-time buyer"
        regulatory_flags.append(
            f"LTV ({ltv_ratio:.1%}) exceeds {buyer} regulatory cap of {max_ltv:.0%}"
        )
    if lti_ratio > max_lti:
        regulatory_flags.append(
            f"LTI ({lti_ratio:.2f}x) exceeds regulatory cap of {max_lti}x income"
        )
    if deposit < required_deposit:
        shortfall = required_deposit - deposit
        regulatory_flags.append(
            f"Deposit (EUR {deposit:,.0f}) is EUR {shortfall:,.0f} short of the "
            f"EUR {required_deposit:,.0f} equity gap"
        )
    elif deposit < min_deposit_for_ltv_cap:
        regulatory_flags.append(
            f"Deposit (EUR {deposit:,.0f}) is below the "
            f"EUR {min_deposit_for_ltv_cap:,.0f} minimum for a {max_ltv:.0%} LTV cap"
        )

    if dti_ratio > 0.40:
        warnings.append(
            f"DTI ({dti_ratio:.1%}) is high — above typical 40% affordability threshold"
        )
    if stressed_dti_ratio > 0.40:
        warnings.append(
            f"Stress-tested DTI at +{stress_bps} bps ({stressed_dti_ratio:.1%}) "
            f"exceeds 40% affordability threshold"
        )
    if net_disposable_surplus < 0:
        warnings.append(
            f"Estimated net disposable surplus is negative "
            f"(EUR {net_disposable_surplus:,.2f}/month)"
        )
    elif stressed_net_disposable_surplus < 0:
        warnings.append(
            f"Stress-tested estimated net disposable surplus is negative "
            f"(EUR {stressed_net_disposable_surplus:,.2f}/month at +{stress_bps} bps)"
        )

    flags = regulatory_flags + warnings

    return {
        "monthly_repayment": round(monthly_repayment, 2),
        "stressed_monthly_repayment": round(stressed_monthly_repayment, 2),
        "stressed_interest_rate": round(stressed_rate, 2),
        "stress_bps": stress_bps,
        "dti_ratio": round(dti_ratio, 4),
        "stressed_dti_ratio": round(stressed_dti_ratio, 4),
        "ltv_ratio": round(ltv_ratio, 4),
        "lti_ratio": round(lti_ratio, 2),
        "max_ltv": max_ltv,
        "max_lti": max_lti,
        "first_time_buyer": first_time_buyer,
        "deposit": round(float(deposit), 2),
        "required_deposit": round(float(required_deposit), 2),
        "deposit_sufficient": deposit >= required_deposit
        and deposit >= min_deposit_for_ltv_cap,
        "estimated_net_monthly_income": net_monthly_income,
        "effective_tax_rate": effective_tax_rate,
        "net_disposable_surplus": net_disposable_surplus,
        "stressed_net_disposable_surplus": stressed_net_disposable_surplus,
        "within_regulatory_limits": len(regulatory_flags) == 0,
        "regulatory_flags": regulatory_flags,
        "warnings": warnings,
        "flags": flags,
    }
