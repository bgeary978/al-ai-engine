import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from crewai import Agent, Crew, LLM, Task

llm = LLM(
    model="ollama/llama3.1",
    base_url="http://localhost:11434",
)

assistant = Agent(
    role="Irish Mortgage Assistant",
    goal="Explain mortgage concepts clearly for an Irish retail lender",
    backstory=(
        "You work on the AL-AI Engine credit committee. "
        "You explain concepts only. You do not approve loans."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

task = Task(
    description=(
        "In two short sentences, explain what loan-to-value (LTV) means "
        "for an Irish mortgage, and state the typical owner-occupier cap."
    ),
    expected_output="Two sentences a loan officer could read aloud.",
    agent=assistant,
)

crew = Crew(agents=[assistant], tasks=[task], verbose=True)

if __name__ == "__main__":
    print(crew.kickoff())
