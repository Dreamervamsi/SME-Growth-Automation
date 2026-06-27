import os
import pytest
from src.langraph import (
    init_db,
    seed_db,
    get_inventory_levels,
    update_stock,
    get_customers,
    get_customer_segments,
    get_leads,
    get_campaigns,
    add_campaign,
    CustomerProfile,
    InventoryItem,
    TrackedLead,
    ActiveCampaign,
)

TEST_DB_PATH = "test_sme_assistant.db"

@pytest.fixture(autouse=True)
def setup_and_teardown_db():
    # Clean up before test
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except OSError:
            pass
    yield
    # Clean up after test
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except OSError:
            pass


def test_init_and_seed_db():
    # Test initialization
    init_db(TEST_DB_PATH)
    assert os.path.exists(TEST_DB_PATH)

    # Test seeding
    seed_db(TEST_DB_PATH)

    # Validate inventory items
    inventory = get_inventory_levels(TEST_DB_PATH)
    assert len(inventory) == 8
    
    # Verify slow-moving item existence
    slow_items = [item for item in inventory if item["days_in_stock_unsold"] > 90]
    assert len(slow_items) >= 2
    skus_slow = {item["sku"] for item in slow_items}
    assert "SKU-KT-BLU" in skus_slow
    assert "SKU-SD-BRN" in skus_slow

    # Validate Pydantic compatibility
    inventory_models = [InventoryItem(**item) for item in inventory]
    assert len(inventory_models) == 8
    assert any(m.sku == "SKU-KT-BLU" for m in inventory_models)

    # Validate customers
    customers = get_customers(TEST_DB_PATH)
    assert len(customers) == 18
    # Validate Pydantic compatibility
    customer_models = [CustomerProfile(**c) for c in customers]
    assert len(customer_models) == 18
    assert customer_models[0].customer_id == "CUST001"
    assert customer_models[0].name == "Kavya Sharma"
    assert customer_models[0].total_spend == 5100.0

    # Validate leads
    leads = get_leads(TEST_DB_PATH)
    assert len(leads) == 6
    # Validate Pydantic compatibility
    lead_models = [TrackedLead(**l) for l in leads]
    assert len(lead_models) == 6
    assert lead_models[0].lead_id == "LEAD001"
    assert lead_models[0].lead_score == 8.5

    # Validate campaigns
    campaigns = get_campaigns(TEST_DB_PATH)
    assert len(campaigns) == 2
    # Validate Pydantic compatibility
    campaign_models = [ActiveCampaign(**camp) for camp in campaigns]
    assert len(campaign_models) == 2
    assert campaign_models[0].campaign_id == "CAMP001"
    assert campaign_models[0].status == "Sent"


def test_update_stock():
    seed_db(TEST_DB_PATH)
    sku = "SKU-KT-BLU"
    
    # Check current quantity
    inventoryBefore = {item["sku"]: item["stock_quantity"] for item in get_inventory_levels(TEST_DB_PATH)}
    assert inventoryBefore[sku] == 45
    
    # Update quantity
    success = update_stock(sku, 50, TEST_DB_PATH)
    assert success is True
    
    # Check new quantity
    inventoryAfter = {item["sku"]: item["stock_quantity"] for item in get_inventory_levels(TEST_DB_PATH)}
    assert inventoryAfter[sku] == 50

    # Try updating non-existent SKU
    success_fake = update_stock("SKU-NON-EXISTENT", 10, TEST_DB_PATH)
    assert success_fake is False


def test_get_customer_segments():
    seed_db(TEST_DB_PATH)

    # Filter VIPs (spend >= 5000)
    vips = get_customer_segments({"min_spend": 5000.0}, TEST_DB_PATH)
    assert len(vips) == 2
    vip_ids = {c["customer_id"] for c in vips}
    assert "CUST001" in vip_ids
    assert "CUST007" in vip_ids

    # Filter WhatsApp preference
    whatsapp_users = get_customer_segments({"preferred_channel": "whatsapp"}, TEST_DB_PATH)
    assert len(whatsapp_users) > 0
    # Check that preferred_channel in all is whatsapp
    for wu in whatsapp_users:
        assert wu["contact_info"]["preferred_channel"] == "whatsapp"

    # Filter by purchased slow-moving item "SKU-KT-BLU"
    buyers_of_blue_kurta = get_customer_segments({"purchased_item": "SKU-KT-BLU"}, TEST_DB_PATH)
    assert len(buyers_of_blue_kurta) == 2
    buyer_ids = {c["customer_id"] for c in buyers_of_blue_kurta}
    assert "CUST005" in buyer_ids
    assert "CUST011" in buyer_ids

    # Filter by inactivity (days since last purchase >= 120 days)
    inactive = get_customer_segments({"inactive_days_min": 120}, TEST_DB_PATH)
    assert len(inactive) >= 2
    inactive_ids = {c["customer_id"] for c in inactive}
    assert "CUST004" in inactive_ids
    assert "CUST015" in inactive_ids


def test_add_campaign():
    seed_db(TEST_DB_PATH)

    new_campaign = {
        "title": "Summer Blowout",
        "target_segment": "All Customers",
        "marketing_channels": ["sms", "email"],
        "generated_content": {
            "sms": "Summer sale starts now!",
            "email": "Summer blowout details..."
        },
        "status": "Draft",
        "budget": 250.0,
        "estimated_roi": 1.8
    }

    camp_id = add_campaign(new_campaign, TEST_DB_PATH)
    assert camp_id.startswith("CAMP-")

    campaigns = get_campaigns(TEST_DB_PATH)
    assert len(campaigns) == 3
    
    added_camp = [c for c in campaigns if c["campaign_id"] == camp_id][0]
    assert added_camp["title"] == "Summer Blowout"
    assert added_camp["marketing_channels"] == ["sms", "email"]
    assert added_camp["generated_content"]["sms"] == "Summer sale starts now!"
    assert added_camp["budget"] == 250.0
    assert added_camp["status"] == "Draft"
