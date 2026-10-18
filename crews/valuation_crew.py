"""Valuation Agent as a CrewAI agent on a CLOUD LLM (Groq).

Place this file at: al-ai-engine/crews/valuation_crew.py

Groq is reached through its OpenAI-compatible endpoint using CrewAI's native
"openai/" provider. (The LiteLLM route fails with a 'cache_breakpoint' error
on CrewAI 1.15.27.)

To switch cloud provider later, change the LLM(...) block below, e.g.
  Gemini:    LLM(model="gemini/<model-name>")        needs GEMINI_API_KEY
  Anthropic: LLM(model="anthropic/<model-name>")     needs ANTHROPIC_API_KEY
  OpenAI:    LLM(model="openai/<model-name>")        needs OPENAI_API_KEY
Check the current model names in each provider's console.
"""
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")  # loads GROQ_API_KEY

from crewai import Agent, Task, Crew, LLM  # noqa: E402
from tools.al_tools import valuation_tool  # noqa: E402

# Check the exact current model name in the Groq console if this errors.
GROQ_MODEL = "openai/gpt-oss-120b"

cloud_llm = LLM(
    model=f"openai/{GROQ_MODEL}",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1,
)


def build_valuation_crew(prop: dict) -> Crew:
    valuation_agent = Agent(
        role="Property Valuation Analyst",
        goal=(
            "Determine whether a property's asking price is fair, overvalued or "
            "undervalued compared with the ML valuation model, using only the tool."
        ),
        backstory=(
            "You are a careful Irish property valuation analyst. You never "
            "estimate prices yourself. You call the Property Valuation Checker "
            "tool and explain its results accurately."
        ),
        tools=[valuation_tool],
        llm=cloud_llm,
        allow_delegation=False,
        verbose=True,
    )

    task = Task(
        description=(
            "Assess this property by calling the Property Valuation Checker tool "
            "exactly once with these exact values:\n"
            f"- property_type: {prop['property_type']}\n"
            f"- county: {prop['county']}\n"
            f"- ber_rating: {prop['ber_rating']}\n"
            f"- bedrooms: {prop['bedrooms']}\n"
            f"- bathrooms: {prop['bathrooms']}\n"
            f"- floor_area: {prop['floor_area']}\n"
            f"- area_grouped: {prop['area_grouped']}\n"
            f"- asking_price: {prop['asking_price']}\n\n"
            "Do NOT estimate or calculate any figure yourself. Report only "
            "numbers returned by the tool."
        ),
        expected_output=(
            "Return ONLY a JSON object with these keys: predicted_value, "
            "asking_price, percentage_difference, overvalued, undervalued "
            "(all copied exactly from the tool output) and summary (one "
            "plain-English sentence saying whether the asking price is fair, "
            "overvalued or undervalued and by how much). No text outside the JSON."
        ),
        agent=valuation_agent,
    )

    return Crew(agents=[valuation_agent], tasks=[task], verbose=True)
