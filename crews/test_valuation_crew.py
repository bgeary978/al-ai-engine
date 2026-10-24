
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.valuation_agent import assess_valuation  
from tools.al_tools import _get_model  
from crews.valuation_crew import build_valuation_crew  

prop = {
    "property_type": "House",
    "county": "Dublin",
    "ber_rating": "D1",
    "bedrooms": 4,
    "bathrooms": 4,
    "floor_area": 158,
    "area_grouped": "Other",
    "asking_price": 900000,
}

expected = assess_valuation(
    _get_model(),
    prop["property_type"], prop["county"], prop["ber_rating"],
    prop["bedrooms"], prop["bathrooms"], prop["floor_area"],
    prop["area_grouped"], prop["asking_price"],
)
print("\n=== GROUND TRUTH (direct function call) ===")
print(expected)

result = str(build_valuation_crew(prop).kickoff())
print("\n=== AGENT OUTPUT ===")
print(result)

digits = re.sub(r"[,\s]", "", result)
print("\n=== VERIFICATION ===")
pv = f"{expected['predicted_value']:.2f}"
pd_ = f"{expected['percentage_difference']:.2f}"
print(f"predicted_value: {'OK' if pv in digits else 'MISMATCH - agent may have invented it'}")
print(f"percentage_difference: {'OK' if pd_ in digits else 'MISMATCH - agent may have invented it'}")
said_over = "overvalued" in result.lower()
print(f"conclusion (expected overvalued={expected['overvalued']}): "
      f"{'OK' if said_over == expected['overvalued'] else 'CHECK - conclusion may be wrong'}")