
import json
from browser_use.agent.views import AgentOutput
from browser_use.tools.registry.views import ActionModel
from pydantic import BaseModel

# Mocking the dynamic ActionModel creation if needed, but AgentOutput imports it.
# In a real run, ActionModel is populated with actions.
# Let's see what the schema looks like.

try:
    print(json.dumps(AgentOutput.model_json_schema(), indent=2))
except Exception as e:
    print(f"Error: {e}")
