"""Underwriting Strategist Agent on a CLOUD LLM (OpenAI).

Place this file at: al-ai-engine/crews/strategist_crew.py

Needs two lines in al-ai-engine/.env:
  OPENAI_API_KEY=...
  OPENAI_MODEL=...      (a chat model name from your OpenAI account)

The Strategist has NO tools. It reads the outputs of the Affordability and
Valuation agents and decides Approve / Conditionally Approve / Decline, then
writes a credit memo that only uses figures it was given.
"""
import json
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from crewai import Agent, Task, Crew, LLM  # noqa: E402

OPENAI_MODEL = os.getenv("OPENAI_MODEL")
if not os.getenv("OPENAI_API_KEY") or not OPENAI_MODEL:
    raise RuntimeError(
        "Add OPENAI_API_KEY and OPENAI_MODEL to al-ai-engine/.env "
        "(see the instructions that came with this file)."
    )

# No temperature set on purpose: some newer OpenAI models reject it.
cloud_llm = LLM(
    model=f"openai/{OPENAI_MODEL}",
    api_key=os.getenv("OPENAI_API_KEY"),
)

LENDING_POLICY = (
    "LENDING POLICY (apply strictly, in this order):\n"
    "1. If affordability.within_regulatory_limits is false (any flag present) "
    "-> decision is 'Decline'.\n"
    "2. Else if valuation.overvalued is true -> decision is "
    "'Conditionally Approve', with at least one condition about an independent "
    "valuation or renegotiating the price.\n"
    "3. Else -> decision is 'Approve'.\n"
)


def build_strategist_crew(affordability: dict, valuation: dict) -> Crew:
    strategist = Agent(
        role="Underwriting Strategist",
        goal=(
            "Make the final lending decision for a mortgage application and "
            "write an audit-ready credit memorandum."
        ),
        backstory=(
            "You are a senior Irish mortgage underwriter. You follow the lending "
            "policy exactly. You only use figures that appear in the inputs you "
            "are given and never invent or recalculate numbers."
        ),
        llm=cloud_llm,
        allow_delegation=False,
        verbose=True,
    )

    task = Task(
        description=(
            f"{LENDING_POLICY}\n"
            "AFFORDABILITY ASSESSMENT (JSON):\n"
            f"{json.dumps(affordability, indent=2)}\n\n"
            "VALUATION ASSESSMENT (JSON):\n"
            f"{json.dumps(valuation, indent=2)}\n\n"
            "Apply the policy to decide. Then write a credit memo of 3 to 5 "
            "sentences explaining the decision using only figures from the two "
            "assessments above."
        ),
        expected_output=(
            "Return ONLY a JSON object with these keys: decision (exactly one of "
            "'Approve', 'Conditionally Approve', 'Decline'), reasons (list of short "
            "strings), conditions (list of strings, empty if none) and credit_memo "
            "(string). No text outside the JSON."
        ),
        agent=strategist,
    )

    return Crew(agents=[strategist], tasks=[task], verbose=True)