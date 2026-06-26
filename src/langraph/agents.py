import json
from typing import Any, Callable, Dict, List
from langchain_core.messages import SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq

from .state import SMEState

def create_agent_node(llm: BaseChatModel, system_prompt: str) -> Callable[[SMEState], Dict[str, Any]]:

    def node(state: SMEState) -> Dict[str, Any]:
        # Serialize snapshots using model_dump (Pydantic v2) or dict (Pydantic v1)
        def to_dict(obj: Any) -> Any:
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            elif hasattr(obj, "dict"):
                return obj.dict()
            return obj

        customer_profiles = [to_dict(p) for p in state.get("customer_profiles", [])]
        current_inventory = [to_dict(i) for i in state.get("current_inventory", [])]
        tracked_leads = [to_dict(l) for l in state.get("tracked_leads", [])]

        state_snapshot = {
            "customer_profiles": customer_profiles,
            "current_inventory": current_inventory,
            "tracked_leads": tracked_leads
        }

        # Build context prompt including the system prompt and the current state snapshot
        full_system_content = (
            f"{system_prompt}\n\n"
            "--- SYSTEM STATE SNAPSHOT ---\n"
            f"{json.dumps(state_snapshot, indent=2)}\n"
            "-----------------------------\n"
        )

        # Build the message list for the LLM
        messages = [SystemMessage(content=full_system_content)] + state.get("messages", [])
        
        # Invoke the LLM
        response = llm.invoke(messages)

        return {
            "messages": [response],
            "next_agents": ["orchestrator"]
        }

    return node


# --- High-Quality System Prompts ---

CRM_AGENT_PROMPT = """You are the CRM (Customer Relationship Management) Agent.
Your core responsibilities are:
1. Managing customer profiles and their details.
2. Tracking purchase history, preferences, and notes.
3. Conducting targeted customer segmentation based on purchase behavior, total spend, and contact preferences.

You have access to the current system state snapshot, which contains the list of customer profiles under 'customer_profiles'.
Analyze this customer data to identify valuable segments (e.g. VIP spenders, inactive accounts, channel preferences) and maintain up-to-date client records. Ensure your output is insightful and directly helps build stronger customer relationships."""

STOCK_AGENT_PROMPT = """You are the Stock and Inventory Agent.
Your core responsibilities are:
1. Monitoring SKU stock levels and flagging low stock.
2. Comparing current stock quantity against the defined 'reorder_point' for each inventory item.
3. Identifying and flagging slow-moving inventory (items with high 'days_in_stock_unsold').

You have access to the current system state snapshot, which contains the inventory status under 'current_inventory'.
Analyze the stock levels and proactively identify items that require restocking or promotional campaigns to clear slow-moving inventory."""

LEADS_AGENT_PROMPT = """You are the Leads Management Agent.
Your core responsibilities are:
1. Parsing incoming leads from various sources (Web Form, Cold Search, etc.).
2. Calculating lead scores based on parameters like estimated deal size, status, and interest level.
3. Drafting personalized, initial outreach strategies for new or high-scoring leads.

You have access to the current system state snapshot, which contains the list of leads under 'tracked_leads'.
Assess incoming leads, assign or update lead scores, and formulate high-impact next-action recommendations."""

MARKETING_AGENT_PROMPT = """You are the Marketing Campaign Agent.
Your core responsibilities are:
1. Designing promotional and marketing copy (specifically tailored templates for WhatsApp, Email, or other channels).
2. Tailoring copy to specific promotional targets, active campaign themes, or customer segments.
3. Formulating ideas and content for active or draft campaigns to maximize conversion and ROI.

You have access to the current system state snapshot, which contains existing campaign details under 'active_campaigns' (in full state context) and reference client/lead info.
Generate highly engaging, relevant, and personalized copy that aligns with the target segment and channel constraints."""


# --- Node Instantiations ---

import os

# Using llama-3.1-70b-versatile as the default chat model instance.
# We fall back to a dummy API key if not set in the environment to prevent import errors.
api_key = os.getenv("GROQ_API_KEY", "dummy_key_to_allow_import")
llm = ChatGroq(model="llama-3.1-70b-versatile", api_key=api_key)

crm_node = create_agent_node(llm, CRM_AGENT_PROMPT)
stock_node = create_agent_node(llm, STOCK_AGENT_PROMPT)
leads_node = create_agent_node(llm, LEADS_AGENT_PROMPT)
marketing_node = create_agent_node(llm, MARKETING_AGENT_PROMPT)


