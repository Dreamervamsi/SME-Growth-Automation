"""
app.py - SME Growth Automation Engine
Production-grade Streamlit dashboard for the live hackathon demo.
"""

import os
import sys
import uuid
import json

from dotenv import load_dotenv

# Load environment variables first (GROQ_API_KEY)
load_dotenv()

import streamlit as st
import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver

# ---------------------------------------------------------------------------
# Path Setup - ensure we can import `src.langraph` from the project root
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.langraph.core import workflow
from src.langraph.database import (
    init_db,
    seed_db,
    get_inventory_levels,
    get_customers,
    get_leads,
    get_campaigns,
    add_campaign,
    get_db_connection,
)
from src.langraph.state import (
    CustomerProfile,
    InventoryItem,
    TrackedLead,
    ActiveCampaign,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DB_PATH = os.path.join(ROOT_DIR, "sme_assistant.db")

AGENT_LABELS = {
    "orchestrator":    ("🧠", "Orchestrator",    "#7C3AED"),
    "crm_agent":       ("👥", "CRM Agent",        "#0EA5E9"),
    "stock_agent":     ("📦", "Stock Agent",      "#F59E0B"),
    "leads_agent":     ("🎯", "Leads Agent",      "#10B981"),
    "marketing_agent": ("📣", "Marketing Agent",  "#EC4899"),
    "FINISH":          ("✅", "Workflow Complete", "#6EE7B7"),
}

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SME Growth Automation Engine",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS - premium dark-mode design with glassmorphism
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Fira+Code:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0A0A14;
    color: #E2E8F0;
}

.stApp {
    background: linear-gradient(135deg, #0A0A14 0%, #0F0F23 50%, #0A0A14 100%);
    min-height: 100vh;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D0D22 0%, #0F1628 100%) !important;
    border-right: 1px solid rgba(124, 58, 237, 0.3);
}
[data-testid="stSidebar"] * { color: #E2E8F0; }

header[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.5rem; }

.dashboard-header {
    background: linear-gradient(135deg, rgba(124,58,237,0.15) 0%, rgba(14,165,233,0.1) 100%);
    border: 1px solid rgba(124,58,237,0.4);
    border-radius: 20px;
    padding: 32px 40px;
    margin-bottom: 28px;
    backdrop-filter: blur(12px);
    position: relative;
    overflow: hidden;
}
.dashboard-header::before {
    content: '';
    position: absolute;
    top: -50%; left: -20%;
    width: 60%; height: 200%;
    background: radial-gradient(ellipse, rgba(124,58,237,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.dashboard-header h1 {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #A78BFA 0%, #38BDF8 50%, #34D399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 10px 0;
    line-height: 1.1;
}
.dashboard-header p {
    color: #94A3B8;
    font-size: 1.05rem;
    margin: 0;
}

.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 20px 16px;
    text-align: center;
    transition: all 0.3s ease;
    height: 100%;
}
.metric-card:hover {
    border-color: rgba(124,58,237,0.5);
    transform: translateY(-2px);
    background: rgba(124,58,237,0.06);
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: #A78BFA;
    line-height: 1;
}
.metric-label {
    font-size: 0.78rem;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 6px;
}

.trace-card {
    border-radius: 12px;
    padding: 12px 16px;
    margin: 8px 0;
    border-left: 3px solid;
    font-size: 0.83rem;
    line-height: 1.5;
    animation: slideIn 0.3s ease;
}
@keyframes slideIn {
    from { opacity:0; transform: translateX(-10px); }
    to   { opacity:1; transform: translateX(0); }
}

.campaign-approval-card {
    background: linear-gradient(135deg, rgba(236,72,153,0.1) 0%, rgba(124,58,237,0.1) 100%);
    border: 1px solid rgba(236,72,153,0.4);
    border-radius: 16px;
    padding: 24px;
    margin: 12px 0;
}
.campaign-approval-card h4 {
    color: #F472B6;
    margin: 0 0 16px 0;
    font-size: 1.1rem;
}

.sidebar-brand {
    text-align: center;
    padding: 16px 0 24px 0;
    border-bottom: 1px solid rgba(124,58,237,0.25);
    margin-bottom: 20px;
}
.sidebar-brand h2 {
    font-size: 1.1rem;
    font-weight: 700;
    background: linear-gradient(135deg, #A78BFA, #38BDF8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 6px 0 0 0;
}
.sidebar-brand p { color: #475569; font-size: 0.75rem; margin: 2px 0 0 0; }

.stButton > button {
    background: linear-gradient(135deg, #7C3AED, #4F46E5) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.25s ease !important;
}
.stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }

.stTabs [data-baseweb="tab-list"] { background: transparent; gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.05);
    border-radius: 10px;
    color: #94A3B8;
    font-size: 0.88rem;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(124,58,237,0.35), rgba(79,70,229,0.35)) !important;
    color: #A78BFA !important;
}

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(124,58,237,0.4); border-radius: 3px; }

[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 14px !important;
    margin-bottom: 8px !important;
}

.stInfo { background: rgba(14,165,233,0.1) !important; border-color: rgba(14,165,233,0.3) !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Database auto-initialization
# ---------------------------------------------------------------------------
@st.cache_resource
def ensure_database():
    """Initialize and seed the DB if it does not exist or is empty."""
    needs_seed = False
    if not os.path.exists(DB_PATH):
        needs_seed = True
    else:
        try:
            with get_db_connection(DB_PATH) as conn:
                count = conn.execute("SELECT COUNT(*) FROM inventory").fetchone()[0]
                if count == 0:
                    needs_seed = True
        except Exception:
            needs_seed = True

    if needs_seed:
        init_db(DB_PATH)
        seed_db(DB_PATH)
        return "seeded"
    return "ready"


# ---------------------------------------------------------------------------
# LangGraph app with MemorySaver + interrupt_after marketing_agent
# ---------------------------------------------------------------------------
@st.cache_resource
def build_graph():
    memory = MemorySaver()
    return workflow.compile(
        checkpointer=memory,
        interrupt_after=["marketing_agent"],
    )


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
def init_session():
    defaults = {
        "messages":          [],
        "thread_id":         str(uuid.uuid4()),
        "execution_trace":   [],
        "awaiting_approval": False,
        "pending_campaign":  None,
        "graph_interrupted": False,
        "db_status":         None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_db_state():
    """Load all DB tables into Pydantic model lists for SMEState."""
    inventory = [InventoryItem(**r) for r in get_inventory_levels(DB_PATH)]
    customers = [CustomerProfile(**r) for r in get_customers(DB_PATH)]
    leads     = [TrackedLead(**r) for r in get_leads(DB_PATH)]
    campaigns = [ActiveCampaign(**r) for r in get_campaigns(DB_PATH)]
    return inventory, customers, leads, campaigns


def add_trace(agent_key: str, reasoning: str = ""):
    emoji, label, color = AGENT_LABELS.get(
        agent_key, ("⚙️", agent_key.replace("_", " ").title(), "#64748B")
    )
    st.session_state.execution_trace.append({
        "agent":     agent_key,
        "emoji":     emoji,
        "label":     label,
        "color":     color,
        "reasoning": reasoning,
    })


def extract_campaign_from_message(content: str) -> dict:
    """Build a minimal campaign dict from the marketing agent's reply."""
    return {
        "title":              "AI Generated Campaign Draft",
        "target_segment":     "Identified Customer Segment",
        "marketing_channels": ["whatsapp", "email"],
        "generated_content":  {"whatsapp": content, "email": content},
        "status":             "Pending Approval",
        "budget":             500.0,
        "estimated_roi":      2.5,
    }


# ---------------------------------------------------------------------------
# Graph runner
# ---------------------------------------------------------------------------
def run_graph(graph, user_message: str):
    """Stream graph execution, logging each node. Returns (last_ai_content, interrupted)."""
    inventory, customers, leads, campaigns = load_db_state()

    initial_state = {
        "messages":          [HumanMessage(content=user_message)],
        "customer_profiles": customers,
        "current_inventory": inventory,
        "tracked_leads":     leads,
        "active_campaigns":  campaigns,
        "next_agents":       [],
        "routing_reasoning": "",
    }
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    last_ai_content  = ""
    seen_nodes       = set()

    try:
        for event in graph.stream(initial_state, config=config, stream_mode="values"):
            routing_reasoning = event.get("routing_reasoning", "")
            next_agents       = event.get("next_agents", [])
            messages          = event.get("messages", [])

            # Log orchestrator when it sets routing
            if routing_reasoning and "orchestrator" not in seen_nodes:
                add_trace("orchestrator", routing_reasoning)
                seen_nodes.add("orchestrator_first")

            # Log orchestrator subsequent calls
            if routing_reasoning and routing_reasoning not in [
                t["reasoning"] for t in st.session_state.execution_trace
            ]:
                add_trace("orchestrator", routing_reasoning)

            # Log specialist agents by detecting new AI messages
            for msg in messages:
                if isinstance(msg, AIMessage) and msg.content and msg.content != last_ai_content:
                    last_ai_content = msg.content
                    # Guess which specialist produced it based on trace order
                    for candidate in ["crm_agent", "stock_agent", "leads_agent", "marketing_agent"]:
                        if candidate not in seen_nodes:
                            add_trace(candidate)
                            seen_nodes.add(candidate)
                            break

        # Check if graph is paused at an interrupt
        state = graph.get_state(config)
        interrupted = bool(state.next)

        return last_ai_content, interrupted

    except Exception as exc:
        st.error(f"⚠️ Graph execution error: {exc}")
        return "", False


def resume_graph(graph) -> str:
    """Resume the graph after HITL approval."""
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    last_content = ""
    try:
        for event in graph.stream(None, config=config, stream_mode="values"):
            next_agents = event.get("next_agents", [])
            messages    = event.get("messages", [])
            if "FINISH" in (next_agents or []):
                add_trace("FINISH", "All requested tasks completed successfully.")
            for msg in messages:
                if isinstance(msg, AIMessage) and msg.content:
                    last_content = msg.content
    except Exception as exc:
        st.error(f"⚠️ Resume error: {exc}")
    return last_content


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand">
            <div style="font-size:2.4rem;">🚀</div>
            <h2>SME Growth Engine</h2>
            <p>Multi-Agent Automation System</p>
        </div>
        """, unsafe_allow_html=True)

        status_icon = "🟢" if st.session_state.db_status == "ready" else "🟡"
        st.caption(f"{status_icon} Database: {st.session_state.db_status or 'Initializing...'}")

        with st.expander("🗄️ Database Controls", expanded=False):
            if st.button("🔄 Re-seed Database", use_container_width=True, key="reseed_btn"):
                with st.spinner("Re-seeding database…"):
                    seed_db(DB_PATH)
                st.success("✅ Database re-seeded with fresh demo data!")
                st.rerun()
            if st.button("🗑️ Reset Conversation", use_container_width=True, key="reset_btn"):
                keys_to_clear = [
                    "messages", "execution_trace", "awaiting_approval",
                    "pending_campaign", "graph_interrupted",
                ]
                for k in keys_to_clear:
                    st.session_state[k] = [] if k in ("messages", "execution_trace") else False if isinstance(st.session_state.get(k), bool) else None
                st.session_state.thread_id = str(uuid.uuid4())
                st.rerun()

        st.divider()

        # ── Execution Trace Panel ──────────────────────────────────────────
        st.markdown("### 🔍 Agent Execution Trace")
        st.markdown(
            "<p style='color:#475569;font-size:0.78rem;margin-bottom:12px;'>"
            "Live reasoning from the Orchestrator and active specialist agents.</p>",
            unsafe_allow_html=True,
        )

        if not st.session_state.execution_trace:
            st.markdown(
                "<p style='color:#334155;font-size:0.83rem;font-style:italic;text-align:center;"
                "padding:16px;border:1px dashed rgba(255,255,255,0.08);border-radius:10px;'>"
                "⏳ Awaiting first request…</p>",
                unsafe_allow_html=True,
            )
        else:
            for step in st.session_state.execution_trace:
                bg     = f"{step['color']}18"
                border = step["color"]
                reasoning_html = ""
                if step["reasoning"]:
                    snippet = step["reasoning"][:240]
                    if len(step["reasoning"]) > 240:
                        snippet += "…"
                    reasoning_html = (
                        f'<div style="color:#94A3B8;font-size:0.78rem;'
                        f'font-family:\'Fira Code\',monospace;margin-top:6px;'
                        f'white-space:pre-wrap;word-break:break-word;">{snippet}</div>'
                    )
                st.markdown(
                    f'<div class="trace-card" style="background:{bg};border-left-color:{border};">'
                    f'  <strong style="color:{border};">{step["emoji"]} {step["label"]}</strong>'
                    f'  {reasoning_html}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        if st.session_state.awaiting_approval:
            st.markdown("""
            <div style="background:rgba(236,72,153,0.12);border:1px solid rgba(236,72,153,0.45);
                        border-radius:12px;padding:14px;margin-top:16px;text-align:center;">
                <span style="color:#F472B6;font-weight:600;font-size:0.9rem;">
                    ⏸️ Paused — Awaiting Approval
                </span>
            </div>
            """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Live Data Viewers
# ---------------------------------------------------------------------------
def render_data_viewers():
    st.markdown("### 📊 Live Database Viewers")
    tab_inv, tab_cust, tab_leads, tab_camps = st.tabs([
        "📦  Inventory", "👥  Customers", "🎯  Leads", "📣  Campaigns",
    ])

    # ── Inventory ──────────────────────────────────────────────────────────
    with tab_inv:
        rows = get_inventory_levels(DB_PATH)
        if rows:
            df = pd.DataFrame(rows)

            def stock_badge(row):
                if row["stock_quantity"] < row["reorder_point"]:
                    return "🔴 Low Stock"
                if row["days_in_stock_unsold"] > 90:
                    return "🟡 Slow Moving"
                return "🟢 OK"

            df["Alert"] = df.apply(stock_badge, axis=1)
            df["Price (₹)"] = df["price"].apply(lambda x: f"₹{x:,.0f}")
            display = df[[
                "sku", "product_name", "category",
                "stock_quantity", "reorder_point", "days_in_stock_unsold",
                "Price (₹)", "Alert",
            ]].rename(columns={
                "sku": "SKU", "product_name": "Product", "category": "Category",
                "stock_quantity": "Stock Qty", "reorder_point": "Reorder Pt",
                "days_in_stock_unsold": "Days Unsold",
            })
            st.dataframe(display, use_container_width=True, hide_index=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Total SKUs", len(rows))
            c2.metric("⚠️ Low Stock Items", sum(1 for r in rows if r["stock_quantity"] < r["reorder_point"]))
            c3.metric("🐢 Slow Moving (>90d)", sum(1 for r in rows if r["days_in_stock_unsold"] > 90))
        else:
            st.info("No inventory data found.")

    # ── Customers ──────────────────────────────────────────────────────────
    with tab_cust:
        rows = get_customers(DB_PATH)
        if rows:
            records = [{
                "ID":            r["customer_id"],
                "Name":          r["name"],
                "Channel":       r["contact_info"].get("preferred_channel", "—").title(),
                "Total Spend":   f"₹{r['total_spend']:,.0f}",
                "Last Purchase": r["last_purchase_date"] or "Never",
                "Notes":         (r["notes"] or "")[:70] + ("…" if r.get("notes") and len(r["notes"]) > 70 else ""),
            } for r in rows]
            st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)

            vips = sum(1 for r in rows if r["total_spend"] >= 5000)
            c1, c2 = st.columns(2)
            c1.metric("Total Customers", len(rows))
            c2.metric("👑 VIP Customers (≥₹5k)", vips)
        else:
            st.info("No customer data found.")

    # ── Leads ──────────────────────────────────────────────────────────────
    with tab_leads:
        rows = get_leads(DB_PATH)
        if rows:
            records = [{
                "Lead ID":    r["lead_id"],
                "Company":    r["company_name"] or "—",
                "Contact":    r["contact_name"] or "—",
                "Source":     r["source"],
                "Deal Size":  f"₹{r['estimated_deal_size']:,.0f}",
                "Score":      r["lead_score"],
                "Status":     r["status"],
                "Next Action": (r["next_action"] or "")[:60] + ("…" if r.get("next_action") and len(r["next_action"]) > 60 else ""),
            } for r in rows]
            st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)

            hot = sum(1 for r in rows if r["lead_score"] >= 8.0)
            c1, c2 = st.columns(2)
            c1.metric("Total Leads", len(rows))
            c2.metric("🔥 Hot Leads (Score ≥ 8)", hot)
        else:
            st.info("No leads data found.")

    # ── Campaigns ──────────────────────────────────────────────────────────
    with tab_camps:
        rows = get_campaigns(DB_PATH)
        if rows:
            records = [{
                "ID":       r["campaign_id"],
                "Title":    r["title"],
                "Segment":  r["target_segment"],
                "Channels": ", ".join(r.get("marketing_channels", [])).upper(),
                "Budget":   f"₹{r['budget']:,.0f}",
                "ROI (est.)": f"{r['estimated_roi']}x" if r.get("estimated_roi") else "—",
                "Conversions": r.get("conversion_count") or 0,
                "Status":   r["status"],
            } for r in rows]
            st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)

            sent  = sum(1 for r in rows if r["status"] == "Sent")
            draft = sum(1 for r in rows if r["status"] == "Draft")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Campaigns", len(rows))
            c2.metric("✅ Sent",  sent)
            c3.metric("📝 Draft", draft)
        else:
            st.info("No campaign data found.")


# ---------------------------------------------------------------------------
# Chat helpers
# ---------------------------------------------------------------------------
def display_chat_history():
    for msg in st.session_state.messages:
        role    = msg.get("role", "assistant")
        content = msg.get("content", "")
        avatar  = "👤" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar):
            st.markdown(content)


def add_user_message(content: str):
    st.session_state.messages.append({"role": "user", "content": content})


def add_assistant_message(content: str):
    st.session_state.messages.append({"role": "assistant", "content": content})


# ---------------------------------------------------------------------------
# HITL Campaign Approval Card
# ---------------------------------------------------------------------------
def render_campaign_approval(graph):
    campaign = st.session_state.pending_campaign
    if not campaign:
        return

    with st.chat_message("assistant", avatar="📣"):
        st.markdown("""
        <div class="campaign-approval-card">
            <h4>📣 Marketing Campaign Draft — Awaiting Your Approval</h4>
        """, unsafe_allow_html=True)

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown(f"**📌 Title:** {campaign.get('title', '—')}")
            st.markdown(f"**🎯 Target Segment:** {campaign.get('target_segment', '—')}")
            st.markdown(f"**📡 Channels:** `{', '.join(campaign.get('marketing_channels', [])).upper()}`")
        with col_r:
            st.markdown(f"**💰 Budget:** ₹{campaign.get('budget', 0):,.0f}")
            st.markdown(f"**📈 Est. ROI:** {campaign.get('estimated_roi', '—')}x")
            st.markdown(f"**🔖 Status:** `{campaign.get('status', 'Pending Approval')}`")

        content = campaign.get("generated_content", {})
        if content:
            st.markdown("---")
            st.markdown("**📝 Generated Content Previews:**")
            for channel, text in content.items():
                with st.expander(f"  {channel.upper()} Template", expanded=False):
                    st.code(text[:600] + ("…" if len(text) > 600 else ""), language=None)

        st.markdown("</div>", unsafe_allow_html=True)

        approve_col, reject_col, spacer = st.columns([1.5, 1.5, 4])
        with approve_col:
            if st.button(
                "✅ Approve & Send",
                key="approve_campaign_btn",
                type="primary",
                use_container_width=True,
            ):
                _handle_approval(graph, approved=True)

        with reject_col:
            if st.button(
                "❌ Reject Draft",
                key="reject_campaign_btn",
                use_container_width=True,
            ):
                _handle_approval(graph, approved=False)


def _handle_approval(graph, approved: bool):
    campaign = st.session_state.pending_campaign

    if approved and campaign:
        campaign["status"] = "Sent"
        camp_id = add_campaign(campaign, DB_PATH)
        add_trace("FINISH", f"Campaign '{campaign['title']}' approved → saved as {camp_id} (Sent).")
        add_assistant_message(
            f"✅ **Campaign approved and sent!**\n\n"
            f"Saved to database as `{camp_id}` with status **Sent**.\n"
            f"Resuming the workflow to finalize all remaining tasks…"
        )
    else:
        add_trace("FINISH", "Campaign draft was rejected by the user.")
        add_assistant_message(
            "❌ **Campaign rejected.** The draft has been discarded.\n"
            "Feel free to submit a new request whenever you're ready."
        )

    # Resume graph
    with st.spinner("⚡ Resuming multi-agent workflow…"):
        final_content = resume_graph(graph)

    if final_content and approved:
        add_assistant_message(
            "🎉 **Workflow complete!** All multi-agent tasks have been executed successfully.\n\n"
            "Check the **Campaigns** tab above to see your new campaign in the live database."
        )

    # Clear approval state
    st.session_state.awaiting_approval = False
    st.session_state.pending_campaign  = None
    st.session_state.graph_interrupted = False
    st.rerun()


# ---------------------------------------------------------------------------
# Main chat input handler
# ---------------------------------------------------------------------------
def handle_user_input(graph, user_input: str):
    add_user_message(user_input)

    # Reset state for new conversation
    st.session_state.execution_trace   = []
    st.session_state.awaiting_approval = False
    st.session_state.pending_campaign  = None
    st.session_state.graph_interrupted = False
    st.session_state.thread_id         = str(uuid.uuid4())

    # Display the user message immediately
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🧠 Multi-agent system is processing your request…"):
            ai_content, interrupted = run_graph(graph, user_input)

        if interrupted and ai_content:
            # Marketing agent paused → HITL
            campaign_draft = extract_campaign_from_message(ai_content)
            st.session_state.pending_campaign  = campaign_draft
            st.session_state.awaiting_approval = True
            st.session_state.graph_interrupted = True

            response_text = (
                "The **Marketing Agent** has analysed your request and drafted a promotional campaign.\n\n"
                "📋 Please review the campaign details below, then **Approve** to save and send it, "
                "or **Reject** to discard the draft."
            )
            st.markdown(response_text)
            add_assistant_message(response_text)

        elif ai_content:
            add_trace("FINISH", "All tasks completed.")
            st.markdown(ai_content)
            add_assistant_message(ai_content)
        else:
            fallback = "The agents have finished processing your request. Check the data viewers above for any updates."
            st.markdown(fallback)
            add_assistant_message(fallback)

    st.rerun()


# ---------------------------------------------------------------------------
# App Entry Point
# ---------------------------------------------------------------------------
def main():
    init_session()

    # Ensure DB is bootstrapped
    db_result = ensure_database()
    st.session_state.db_status = db_result

    # Build LangGraph app
    graph = build_graph()

    # ── Header ───────────────────────────────────────────────────────────
    st.markdown("""
    <div class="dashboard-header">
        <h1>🚀 SME Growth Automation Engine</h1>
        <p>
            AI-powered multi-agent orchestration &mdash; Inventory Management, CRM,
            Lead Intelligence, and Marketing Campaign Generation. Powered by LangGraph &amp; Groq.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Metrics Bar ──────────────────────────────────────────────────────
    inv_rows   = get_inventory_levels(DB_PATH)
    cust_rows  = get_customers(DB_PATH)
    lead_rows  = get_leads(DB_PATH)
    camp_rows  = get_campaigns(DB_PATH)

    low_stock   = sum(1 for r in inv_rows  if r["stock_quantity"] < r["reorder_point"])
    hot_leads   = sum(1 for r in lead_rows if r["lead_score"] >= 8.0)
    camps_sent  = sum(1 for r in camp_rows if r["status"] == "Sent")
    vip_custs   = sum(1 for r in cust_rows if r["total_spend"] >= 5000)

    cols = st.columns(5)
    metrics = [
        (len(inv_rows),  "📦 Total SKUs"),
        (low_stock,      "⚠️ Low Stock"),
        (vip_custs,      "👑 VIP Customers"),
        (hot_leads,      "🔥 Hot Leads"),
        (camps_sent,     "📣 Campaigns Sent"),
    ]
    for col, (val, lbl) in zip(cols, metrics):
        col.markdown(
            f'<div class="metric-card">'
            f'  <div class="metric-value">{val}</div>'
            f'  <div class="metric-label">{lbl}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────
    render_sidebar()

    # ── Live Data Viewers ─────────────────────────────────────────────────
    render_data_viewers()

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    # ── Chat Interface ────────────────────────────────────────────────────
    st.markdown("### 💬 Chat with Your AI Business Assistant")
    st.markdown(
        "<p style='color:#64748B;font-size:0.9rem;margin-bottom:20px;'>"
        "Describe any business challenge and the Orchestrator will automatically route it "
        "across the right specialist agents — all in real time.</p>",
        unsafe_allow_html=True,
    )

    with st.expander("💡 Example Prompts to Try", expanded=False):
        st.markdown("""
| Prompt | Agents Triggered |
|--------|-----------------|
| *"Sales are slow, help me out."* | Stock → Marketing |
| *"Check inventory and create a promotional campaign for slow-moving items."* | Stock → Marketing |
| *"Who are our VIP customers? Draft a WhatsApp campaign for them."* | CRM → Marketing |
| *"Score our pipeline leads and suggest outreach strategies."* | Leads → Marketing |
| *"Give me a full business health check."* | All Agents |
        """)

    # Render chat history
    display_chat_history()

    # Render HITL approval card if the graph is paused
    if st.session_state.awaiting_approval and st.session_state.pending_campaign:
        render_campaign_approval(graph)
        st.info(
            "⏸️ **Chat is paused** — please approve or reject the campaign draft above to continue.",
            icon="⏸️",
        )
    else:
        # Normal chat input
        user_input = st.chat_input(
            "Ask your AI business assistant… e.g. 'Sales are slow, help me out'",
            key="main_chat_input",
        )
        if user_input:
            handle_user_input(graph, user_input)


if __name__ == "__main__":
    main()
