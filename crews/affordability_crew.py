
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crewai import Agent, Task, Crew, LLM  
from tools.al_tools import affordability_tool

local_llm = LLM(
    model="ollama/llama3.1",
    base_url="http://localhost:11434",
    temperature=0.1,  
)


def build_affordability_crew(applicant: dict) -> Crew:
    affordability_agent = Agent(
        role="Affordability Analyst",
        goal=(
            "Assess whether a mortgage applicant can afford the requested loan "
            "under Central Bank of Ireland rules, using only the calculator tool."
        ),
        backstory=(
            "You are a careful Irish mortgage affordability analyst. You never "
            "do arithmetic yourself. You call the Affordability Calculator tool "
            "and explain its results accurately."
        ),
        tools=[affordability_tool],
        llm=local_llm,
        allow_delegation=False,
        verbose=True,
    )

    task = Task(
        description=(
            "Assess this applicant by calling the Affordability Calculator tool "
            "exactly once with these exact values:\n"
            f"- gross_annual_income: {applicant['gross_annual_income']}\n"
            f"- existing_monthly_debts: {applicant['existing_monthly_debts']}\n"
            f"- deposit: {applicant['deposit']}\n"
            f"- property_value: {applicant['property_value']}\n"
            f"- requested_loan_amount: {applicant['requested_loan_amount']}\n"
            f"- interest_rate: {applicant['interest_rate']}\n"
            f"- term_years: {applicant['term_years']}\n"
            f"- first_time_buyer: {applicant['first_time_buyer']}\n\n"
            "Do NOT calculate or estimate any figure yourself. Report only "
            "numbers returned by the tool."
        ),
        expected_output=(
            "A short summary containing the monthly repayment, DTI, LTV and LTI "
            "ratios exactly as returned by the tool, whether the applicant is "
            "within regulatory limits, and any flags."
        ),
        agent=affordability_agent,
    )

    return Crew(agents=[affordability_agent], tasks=[task], verbose=True)