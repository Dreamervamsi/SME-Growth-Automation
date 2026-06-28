import json
import os
from typing import Any, Callable, Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.config import get_stream_writer
from pydantic import BaseModel, Field

from .state import SMEState


def _chunk_text(chunk: Any) -> str:
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content) if content else ""


def _safe_stream_writer() -> Optional[Callable[[Dict[str, Any]], None]]:
    try:
        return get_stream_writer()
    except Exception:
        return None


def _stream_llm_text(
    llm: BaseChatModel,
    messages: List[Any],
    *,
    source: str,
    status_label: Optional[str] = None,
) -> str:
    writer = _safe_stream_writer()
    if writer and status_label:
        writer({"kind": "agent_status", "agent": source, "label": status_label, "status": "started"})

    parts: List[str] = []
    for chunk in llm.stream(messages):
        token = _chunk_text(chunk)
        if token:
            parts.append(token)
            if writer:
                writer({"kind": "token", "source": source, "content": token})

    if writer:
        writer({"kind": "agent_status", "agent": source, "status": "completed"})
    return "".join(parts)


def create_agent_node(
    llm: BaseChatModel,
    system_prompt: str,
    agent_key: str,
) -> Callable[[SMEState], Dict[str, Any]]:
    agent_labels = {
        "crm_agent": "CRM Agent analysing customers",
        "stock_agent": "Stock Agent checking inventory",
        "leads_agent": "Leads Agent scoring pipeline",
        "marketing_agent": "Marketing Agent drafting campaign",
    }

    def node(state: SMEState) -> Dict[str, Any]:
        def to_dict(obj: Any) -> Any:
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            if hasattr(obj, "dict"):
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

        full_system_content = (
            f"{system_prompt}\n\n"
            "--- SYSTEM STATE SNAPSHOT ---\n"
            f"{json.dumps(state_snapshot, indent=2)}\n"
            "-----------------------------\n"
        )

        messages = [SystemMessage(content=full_system_content)] + state.get("messages", [])
        content = _stream_llm_text(
            llm,
            messages,
            source=agent_key,
            status_label=agent_labels.get(agent_key, agent_key),
        )

        return {
            "messages": [AIMessage(content=content)],
            "next_agents": ["orchestrator"],
        }

    return node


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

RESPONDER_SYSTEM_PROMPT = """You are the Final Responder for the SME Growth Assistant.
Write a clear, professional summary for the business owner based on the user's request and the specialist agent outputs below.
Highlight key findings, actions taken, and recommended next steps. Keep the tone friendly and concise."""


api_key = os.getenv("GROQ_API_KEY", "dummy_key_to_allow_import")
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=api_key)

crm_node = create_agent_node(llm, CRM_AGENT_PROMPT, "crm_agent")
stock_node = create_agent_node(llm, STOCK_AGENT_PROMPT, "stock_agent")
leads_node = create_agent_node(llm, LEADS_AGENT_PROMPT, "leads_agent")
marketing_node = create_agent_node(llm, MARKETING_AGENT_PROMPT, "marketing_agent")


class RouterOutput(BaseModel):
    next_step: str = Field(
        ...,
        description=(
            "The next step in the workflow. Allowed values: 'crm_agent', 'stock_agent', "
            "'leads_agent', 'marketing_agent', or 'FINAL_RESPONSE'."
        ),
    )
    reasoning: str = Field(
        ...,
        description=(
            "A short explanation of why this routing decision was made based on the "
            "request and current business state."
        ),
    )
    final_reply_to_user: str = Field(
        default="",
        description="Deprecated — leave empty; the responder node streams the final user reply.",
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
- When all requested operations are fully completed, route to 'FINAL_RESPONSE'. Leave 'final_reply_to_user' empty — a dedicated responder will stream the final summary to the user.
- Always return a valid JSON object matching the requested schema fields: 'next_step', 'reasoning', and 'final_reply_to_user'.
"""


def orchestrator_node(state: SMEState) -> Dict[str, Any]:
    writer = _safe_stream_writer()
    if writer:
        writer({"kind": "agent_status", "agent": "orchestrator", "label": "Orchestrator planning next step", "status": "started"})

    structured_llm = llm.with_structured_output(RouterOutput, method="json_mode")

    def to_dict(obj: Any) -> Any:
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "dict"):
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

    full_system_content = (
        f"{ORCHESTRATOR_SYSTEM_PROMPT}\n\n"
        "--- SYSTEM STATE SNAPSHOT ---\n"
        f"{json.dumps(state_snapshot, indent=2)}\n"
        "-----------------------------\n"
    )

    human_messages = [msg for msg in state.get("messages", []) if msg.type == "human"]
    messages = [SystemMessage(content=full_system_content)] + human_messages

    response = structured_llm.invoke(messages)

    if isinstance(response, dict):
        next_step = response.get("next_step", "FINAL_RESPONSE")
        reasoning = response.get("reasoning", "")
    else:
        next_step = getattr(response, "next_step", "FINAL_RESPONSE")
        reasoning = getattr(response, "reasoning", "")

    valid_agents = {"crm_agent", "stock_agent", "leads_agent", "marketing_agent", "FINAL_RESPONSE"}
    if next_step not in valid_agents:
        next_step = "FINAL_RESPONSE"

    if writer:
        writer({"kind": "routing", "agent": "orchestrator", "next_step": next_step, "reasoning": reasoning})
        writer({"kind": "agent_status", "agent": "orchestrator", "status": "completed"})

    return {
        "next_agents": [next_step],
        "routing_reasoning": reasoning,
        "final_reply_to_user": "",
    }


def responder_node(state: SMEState) -> Dict[str, Any]:
    """Stream the final user-facing summary token-by-token."""
    specialist_outputs = [
        msg.content
        for msg in state.get("messages", [])
        if isinstance(msg, AIMessage) and msg.content
    ]
    specialist_context = "\n\n".join(
        f"Specialist output {index + 1}:\n{content}"
        for index, content in enumerate(specialist_outputs)
    )
    human_messages = [msg for msg in state.get("messages", []) if isinstance(msg, HumanMessage)]

    system_content = RESPONDER_SYSTEM_PROMPT
    if specialist_context:
        system_content += f"\n\n--- SPECIALIST OUTPUTS ---\n{specialist_context}\n"

    messages = [SystemMessage(content=system_content)] + human_messages
    final_reply = _stream_llm_text(
        llm,
        messages,
        source="final",
        status_label="Composing final summary",
    )

    if not final_reply.strip():
        final_reply = state.get("final_reply_to_user") or "Workflow completed successfully."

    return {
        "messages": [AIMessage(content=final_reply)],
        "final_reply_to_user": final_reply,
    }
