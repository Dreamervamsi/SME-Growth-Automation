from langgraph.graph import StateGraph
from src.langraph import (
    SMEState,
    CustomerProfile,
    InventoryItem,
    TrackedLead,
    ActiveCampaign,
)

def test_pydantic_models():
    print("Testing Pydantic models validation...")
    # Test CustomerProfile validation
    profile = CustomerProfile(
        customer_id="CUST001",
        name="Vamsi",
        contact_info={"email": "vamsi@example.com", "phone": "1234567890", "preferred_channel": "whatsapp"},
        purchase_history=[{"item_id": "SKU123", "timestamp": "2026-06-26T12:00:00", "amount": 1500.0}],
        total_spend=1500.0,
        last_purchase_date="2026-06-26",
        notes="Prefers WhatsApp",
    )
    assert profile.customer_id == "CUST001"
    assert profile.total_spend == 1500.0

    # Test InventoryItem validation
    item = InventoryItem(
        sku="SKU123",
        product_name="Designer Kurta",
        category="Apparel",
        stock_quantity=10,
        price=1500.0,
        reorder_point=3,
        days_in_stock_unsold=50,
    )
    assert item.sku == "SKU123"
    assert item.days_in_stock_unsold == 50

    # Test TrackedLead validation
    lead = TrackedLead(
        lead_id="LEAD001",
        source="Web Form",
        contact_name="Aditya",
        estimated_deal_size=5000.0,
        lead_score=8.5,
        status="New",
        next_action="Draft outreach message",
    )
    assert lead.lead_id == "LEAD001"
    assert lead.lead_score == 8.5

    # Test ActiveCampaign validation
    campaign = ActiveCampaign(
        campaign_id="CAMP001",
        title="Weekend Sale",
        target_segment="Premium Customers",
        marketing_channels=["whatsapp"],
        generated_content={"whatsapp": "Get 20% off!"},
        status="Draft",
        budget=500.0,
    )
    assert campaign.campaign_id == "CAMP001"
    print("Pydantic models validation successful!")


def test_state_graph_execution():
    print("Testing StateGraph execution with SMEState...")
    # Construct a simple LangGraph state graph using SMEState to verify integration
    workflow = StateGraph(SMEState)
    
    def initial_node(state: SMEState) -> dict:
        return {
            "customer_profiles": [
                CustomerProfile(
                    customer_id="C001",
                    name="John Doe",
                    contact_info={"email": "john@example.com"},
                )
            ],
            "current_inventory": [
                InventoryItem(
                    sku="I001",
                    product_name="Product A",
                    category="Cat A",
                    stock_quantity=10,
                    price=20.0,
                    reorder_point=2,
                    days_in_stock_unsold=1,
                )
            ],
            "tracked_leads": [
                TrackedLead(
                    lead_id="L001",
                    source="Cold Search",
                    contact_name="Sales lead",
                    estimated_deal_size=100.0,
                    lead_score=5.0,
                    status="New",
                )
            ],
            "active_campaigns": [
                ActiveCampaign(
                    campaign_id="CAMP001",
                    title="Campaign One",
                    target_segment="All",
                )
            ],
            "next_agents": ["marketing_agent"],
        }

    def marketing_node(state: SMEState) -> dict:
        # Check current campaigns
        assert len(state["active_campaigns"]) == 1
        assert state["active_campaigns"][0].campaign_id == "CAMP001"
        
        # Append a new campaign, and override next_agents
        return {
            "active_campaigns": [
                ActiveCampaign(
                    campaign_id="CAMP002",
                    title="Campaign Two",
                    target_segment="All",
                )
            ],
            "next_agents": ["crm_agent"],
        }

    workflow.add_node("initial_node", initial_node)
    workflow.add_node("marketing_node", marketing_node)
    
    workflow.set_entry_point("initial_node")
    workflow.add_edge("initial_node", "marketing_node")
    
    app = workflow.compile()
    
    # Run the graph
    final_state = app.invoke({
        "messages": [],
        "customer_profiles": [],
        "current_inventory": [],
        "tracked_leads": [],
        "active_campaigns": [],
        "next_agents": [],
    })
    
    # Verify that:
    # 1. customer_profiles, current_inventory, tracked_leads are set
    assert len(final_state["customer_profiles"]) == 1
    assert final_state["customer_profiles"][0].name == "John Doe"
    
    # 2. active_campaigns were appended (so now we have 2)
    assert len(final_state["active_campaigns"]) == 2
    assert final_state["active_campaigns"][0].campaign_id == "CAMP001"
    assert final_state["active_campaigns"][1].campaign_id == "CAMP002"
    
    # 3. next_agents was overridden and equals ["crm_agent"]
    assert final_state["next_agents"] == ["crm_agent"]
    print("StateGraph execution successful!")

if __name__ == "__main__":
    test_pydantic_models()
    test_state_graph_execution()
    print("All tests passed successfully!")
