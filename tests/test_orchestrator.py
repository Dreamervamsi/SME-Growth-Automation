import pytest
from typing import Any, List
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel

from src.langraph import (
    RouterOutput,
    orchestrator_node,
    SMEState,
    CustomerProfile,
    InventoryItem,
    TrackedLead,
    ActiveCampaign,
    app,
)
from src.langraph.agents import llm

def test_router_output_schema():
    # Verify that RouterOutput validates inputs correctly
    valid_data = {
        "next_agents": ["crm_agent", "marketing_agent"],
        "reasoning": "Need to follow up with VIPs and draft email template."
    }
    router_out = RouterOutput(**valid_data)
    assert router_out.next_agents == ["crm_agent", "marketing_agent"]
    assert router_out.reasoning == "Need to follow up with VIPs and draft email template."


def test_orchestrator_node(monkeypatch):
    # Mock output from structured LLM
    expected_output = RouterOutput(
        next_agents=["stock_agent"],
        reasoning="Low stock levels detected. Checking inventory first."
    )

    class MockStructuredRunnable:
        def invoke(self, messages, *args, **kwargs):
            return expected_output

    # Monkeypatch the with_structured_output method of ChatGroq class
    monkeypatch.setattr(ChatGroq, "with_structured_output", lambda self, schema, **kwargs: MockStructuredRunnable())

    # Prepare input state
    state: SMEState = {
        "messages": [HumanMessage(content="Check if we have enough stock.")],
        "customer_profiles": [],
        "current_inventory": [
            InventoryItem(
                sku="SKU-1",
                product_name="Product A",
                category="Apparel",
                stock_quantity=2,
                price=10.0,
                reorder_point=5,
                days_in_stock_unsold=10
            )
        ],
        "tracked_leads": [],
        "active_campaigns": [],
        "next_agents": [],
        "routing_reasoning": ""
    }

    # Execute orchestrator_node
    result = orchestrator_node(state)

    # Assert results
    assert result["next_agents"] == ["stock_agent"]
    assert result["routing_reasoning"] == "Low stock levels detected. Checking inventory first."


def test_compiled_graph_execution(monkeypatch):
    # Test workflow execution with mock responses
    call_sequence = []

    # Mock the with_structured_output for orchestrator
    # First call: route to stock_agent
    # Second call: route to marketing_agent
    # Third call: route to FINISH
    orchestrator_outputs = [
        RouterOutput(next_agents=["stock_agent"], reasoning="Check stock levels."),
        RouterOutput(next_agents=["marketing_agent"], reasoning="Draft promotion for stock."),
        RouterOutput(next_agents=["FINISH"], reasoning="All tasks complete.")
    ]

    class MockStructuredRunnable:
        def invoke(self, messages, *args, **kwargs):
            call_sequence.append("orchestrator")
            return orchestrator_outputs.pop(0)

    monkeypatch.setattr(ChatGroq, "with_structured_output", lambda self, schema, **kwargs: MockStructuredRunnable())

    # Mock specialists invokes
    # We monkeypatch the invoke of the ChatGroq class
    def mock_llm_invoke(self, messages, *args, **kwargs):
        # Determine which agent was called by looking at the system message content
        system_msg = messages[0].content
        if "CRM" in system_msg:
            call_sequence.append("crm")
            return AIMessage(content="CRM completed.")
        elif "Stock" in system_msg:
            call_sequence.append("stock")
            return AIMessage(content="Stock completed.")
        elif "Leads" in system_msg:
            call_sequence.append("leads")
            return AIMessage(content="Leads completed.")
        elif "Marketing" in system_msg:
            call_sequence.append("marketing")
            return AIMessage(content="Marketing completed.")
        else:
            call_sequence.append("unknown")
            return AIMessage(content="Unknown completed.")

    monkeypatch.setattr(ChatGroq, "invoke", mock_llm_invoke)

    # Initialize graph state
    initial_state = {
        "messages": [HumanMessage(content="Run low stock promotion workflow.")],
        "customer_profiles": [],
        "current_inventory": [],
        "tracked_leads": [],
        "active_campaigns": [],
        "next_agents": [],
        "routing_reasoning": ""
    }

    # Run the compiled graph
    final_state = app.invoke(initial_state)

    # Check execution trace
    # Sequence should be:
    # 1. Orchestrator runs, decides stock_agent
    # 2. Stock agent runs, routes back to Orchestrator
    # 3. Orchestrator runs, decides marketing_agent
    # 4. Marketing agent runs, routes back to Orchestrator
    # 5. Orchestrator runs, decides FINISH
    # 6. Graph terminates
    assert call_sequence == ["orchestrator", "stock", "orchestrator", "marketing", "orchestrator"]
    assert final_state["next_agents"] == ["FINISH"]
    assert final_state["routing_reasoning"] == "All tasks complete."
    assert len(final_state["messages"]) >= 3 # original human message + 2 AI messages from agents
