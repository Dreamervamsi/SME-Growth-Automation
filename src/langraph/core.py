from typing import Any, Dict, List, Union
from langgraph.graph import StateGraph, END
from .state import SMEState
from .agents import (
    crm_node,
    stock_node,
    leads_node,
    marketing_node,
    orchestrator_node,
)
class Langraph:
    """Simple base class for the langraph package."""
    def __init__(self, name: str = "langraph") -> None:
        self.name = name
    def describe(self) -> str:
        return f"Langraph package initialized for {self.name}"
def route_agents(state: SMEState) -> Union[str, List[str]]:
    """Determine the next step in the workflow based on state['next_agents']."""
    next_agents = state.get("next_agents", [])
    if not next_agents or "FINISH" in next_agents:
        return "FINISH"
    return next_agents
# Define workflow pipeline
workflow = StateGraph(SMEState)
# Add all 5 nodes
workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("crm_agent", crm_node)
workflow.add_node("stock_agent", stock_node)
workflow.add_node("leads_agent", leads_node)
workflow.add_node("marketing_agent", marketing_node)
# Set entry point
workflow.set_entry_point("orchestrator")
# Implement conditional edge from the orchestrator
workflow.add_conditional_edges(
    "orchestrator",
    route_agents,
    {
        "crm_agent": "crm_agent",
        "stock_agent": "stock_agent",
        "leads_agent": "leads_agent",
        "marketing_agent": "marketing_agent",
        "FINISH": END,
    }
)
# Add normal edges from specialist agents back to orchestrator to maintain the supervisor loop
workflow.add_edge("crm_agent", "orchestrator")
workflow.add_edge("stock_agent", "orchestrator")
workflow.add_edge("leads_agent", "orchestrator")
workflow.add_edge("marketing_agent", "orchestrator")
# Compile the workflow graph
app = workflow.compile()