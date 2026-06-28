import json
from typing import Any, Callable, Dict, List
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from .state import SMEState

def create_agent_node(llm: BaseChatModel, system_prompt: str) -> Callable[[SMEState], Dict[str, Any]]:

    def node(state: SMEState) -> Dict[str, Any]:
        def to_dict(obj: Any) -> Any:
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            elif hasattr(obj, "dict"):
                return obj.dict()
            return obj

        customer_profiles = [to_dict(p) for p in state.get("customer_profiles", [])]
        current_inventory = [to_dict(i) for i in state.get("current_inventory", [])]
        tracked_leads = [to_dict(l) for l in state.get("tracked_leads", [])]
        active_campaigns = [to_dict(c) for c in state.get("active_campaigns", [])]

        state_snapshot = {
            "customer_profiles": customer_profiles,
            "current_inventory": current_inventory,
            "tracked_leads": tracked_leads,
            "active_campaigns": active_campaigns,
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

# System Prompts

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

# Fall back to a placeholder key at import time so the module loads without .env;
# set GROQ_API_KEY in the environment before invoking the LLM.
api_key = os.getenv("GROQ_API_KEY", "dummy_key_to_allow_import")
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=api_key)

crm_node = create_agent_node(llm, CRM_AGENT_PROMPT)
stock_node = create_agent_node(llm, STOCK_AGENT_PROMPT)
leads_node = create_agent_node(llm, LEADS_AGENT_PROMPT)
marketing_node = create_agent_node(llm, MARKETING_AGENT_PROMPT)


# --- Orchestrator / Supervisor Routing ---

class RouterOutput(BaseModel):
    next_step: str = Field(
        ...,
        description="The next step in the workflow. Allowed values: 'crm_agent', 'stock_agent', 'leads_agent', 'marketing_agent', or 'FINAL_RESPONSE'."
    )
    reasoning: str = Field(
        ...,
        description="A short explanation of why this routing decision was made based on the request and current business state."
    )
    final_reply_to_user: str = Field(
        default="",
        description="A friendly, clear conversational summary for the business owner explaining what was done, key findings, and recommended next steps. Populate ONLY when next_step is 'FINAL_RESPONSE'."
    )


ORCHESTRATOR_SYSTEM_PROMPT = """You are the Orchestrator (Supervisor) Agent for the SME Growth Assistant.
Your core responsibility is to analyze the user's request and the current business state snapshot to determine the best specialist agent(s) to route to next, or finish the workflow.

Specialist Agents and their responsibilities:
1. 'crm_agent': Handles customer profile management, VIP analysis, preferences, total spend, and contact details.
2. 'stock_agent': Monitors SKU stock levels, flags low stock (< reorder point), and identifies slow-moving inventory (> 90 days unsold).
3. 'leads_agent': Manages new leads, estimates deal sizes, scores leads, and plans outreach next actions.
4. 'marketing_agent': Designs promotional campaigns and drafts templates/messages (Email/WhatsApp).

Routing Guidelines:
- If a workflow involves multiple tasks, address them sequentially. For example:
  - If inventory is low or unsold for a long time, route to 'stock_agent' first to identify the specific items, then route to 'marketing_agent' or 'crm_agent' to design promotions for those items or update profiles.
  - If a lead needs scoring and then outreach copy, route to 'leads_agent' first, then 'marketing_agent'.
- Once a specialist agent completes its task, it will hand back control to you. Re-evaluate the updated state to decide the next step.
- When all requested operations are fully completed, route to 'FINAL_RESPONSE'.
- In the final step (when next_step is 'FINAL_RESPONSE'), you MUST read the specialists' changes/data in the state snapshot and write a comprehensive, professional, and friendly response to the business owner explaining what was done, key findings, and recommended next steps in the 'final_reply_to_user' field. Do not leave 'final_reply_to_user' blank.
- Always return a valid JSON object matching the requested schema fields: 'next_step', 'reasoning', and 'final_reply_to_user'.
"""


def orchestrator_node(state: SMEState) -> Dict[str, Any]:
    # Force structured output via JSON mode for Groq compatibility
    structured_llm = llm.with_structured_output(RouterOutput, method="json_mode")

    def to_dict(obj: Any) -> Any:
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        elif hasattr(obj, "dict"):
            return obj.dict()
        return obj

    customer_profiles = [to_dict(p) for p in state.get("customer_profiles", [])]
    current_inventory = [to_dict(i) for i in state.get("current_inventory", [])]
    tracked_leads = [to_dict(l) for l in state.get("tracked_leads", [])]
    active_campaigns = [to_dict(c) for c in state.get("active_campaigns", [])]

    state_snapshot = {
        "customer_profiles": customer_profiles,
        "current_inventory": current_inventory,
        "tracked_leads": tracked_leads,
        "active_campaigns": active_campaigns
    }

    full_system_content = (
        f"{ORCHESTRATOR_SYSTEM_PROMPT}\n\n"
        "--- SYSTEM STATE SNAPSHOT ---\n"
        f"{json.dumps(state_snapshot, indent=2)}\n"
        "-----------------------------\n"
    )

    # 🛑 FIX: Filter out intermediate specialist assistant responses.
    # This prevents the JSON parser from choking and stops the model from looping.
    human_messages = [msg for msg in state.get("messages", []) if msg.type == "human"]
    
    # Construct clean payload: System instructions + User Query only
    messages = [SystemMessage(content=full_system_content)] + human_messages

    # Invoke Structured LLM
    response = structured_llm.invoke(messages)

    if isinstance(response, dict):
        next_step = response.get("next_step", "FINAL_RESPONSE")
        reasoning = response.get("reasoning", "")
        final_reply_to_user = response.get("final_reply_to_user", "")
    else:
        next_step = getattr(response, "next_step", "FINAL_RESPONSE")
        reasoning = getattr(response, "reasoning", "")
        final_reply_to_user = getattr(response, "final_reply_to_user", "")

    # Validate next_step values
    valid_agents = {"crm_agent", "stock_agent", "leads_agent", "marketing_agent", "FINAL_RESPONSE"}
    if next_step not in valid_agents:
        next_step = "FINAL_RESPONSE"

    return {
        "next_agents": [next_step],
        "routing_reasoning": reasoning,
        "final_reply_to_user": final_reply_to_user
    }

def responder_node(state: SMEState) -> Dict[str, Any]:
    """
    Final Responder Node: Appends the final summary constructed by the orchestrator
    to the state message history.
    """
    final_reply = state.get("final_reply_to_user", "")
    if not final_reply:
        final_reply = "Workflow completed successfully."
    return {
        "messages": [AIMessage(content=final_reply)]
    }