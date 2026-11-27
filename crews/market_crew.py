import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from crewai import Agent, Task, Crew, LLM
from tools.al_tools import market_research_tool

CREW_VERBOSE = os.getenv("CREW_VERBOSE", "0").lower() in ("1", "true", "yes")

# Second cloud provider alongside OpenAI (strategist) — Groq for market research.
GROQ_MODEL = "gpt-oss-120b"

cloud_llm = LLM(
    model=f"openai/{GROQ_MODEL}",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1,
)


def build_market_crew(county: str) -> Crew:
    researcher = Agent(
        role="Market & Macro Researcher",
        goal=(
            "Pull current central-bank base rates and prevailing Irish mortgage "
            "benchmark trends for the underwriting committee, using only the "
            "Market Macro Researcher tool."
        ),
        backstory=(
            "You are a careful Irish mortgage market researcher. You never invent "
            "rates or trends. You call the Market Macro Researcher tool and "
            "summarise its results accurately for underwriters."
        ),
        tools=[market_research_tool],
        llm=cloud_llm,
        allow_delegation=False,
        verbose=CREW_VERBOSE,
    )

    task = Task(
        description=(
            "Research current mortgage market conditions for this county by "
            f"calling the Market Macro Researcher tool exactly once with county: "
            f"{county}\n\n"
            "Do NOT invent rates, trends or risk flags. Report only figures "
            "returned by the tool, and note whether the data is live or a "
            "static fallback."
        ),
        expected_output=(
            "A short summary of ECB rate, typical variable and fixed mortgage "
            "rates, house-price trend, any risk flags, whether data is live, "
            "and key sources — all taken from the tool output."
        ),
        agent=researcher,
    )

    return Crew(agents=[researcher], tasks=[task], verbose=CREW_VERBOSE)
