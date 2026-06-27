import sqlite3
import json
import os
from contextlib import contextmanager
from typing import List, Dict, Any, Optional

@contextmanager
def get_db_connection(db_path: str = "sme_assistant.db"):
    """
    Context manager that yields an SQLite connection, ensuring it is properly closed.
    Converts query results into dictionary-like sqlite3.Row objects.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: str = "sme_assistant.db") -> None:
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        
        # Customers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS customers (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT,
                phone TEXT,
                preferred_channel TEXT,
                purchase_history TEXT,
                total_spend REAL DEFAULT 0.0,
                last_purchase_date TEXT,
                notes TEXT
            )
        """)
        
        # Inventory table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS inventory (
                sku TEXT PRIMARY KEY,
                product_name TEXT NOT NULL,
                category TEXT,
                stock_quantity INTEGER NOT NULL,
                price REAL NOT NULL,
                reorder_point INTEGER NOT NULL,
                days_in_stock_unsold INTEGER DEFAULT 0
            )
        """)
        
        # Leads table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS leads (
                id TEXT PRIMARY KEY,
                source TEXT,
                company_name TEXT,
                contact_name TEXT,
                estimated_deal_size REAL DEFAULT 0.0,
                lead_score REAL DEFAULT 0.0,
                status TEXT DEFAULT 'New',
                next_action TEXT
            )
        """)
        
        # Campaigns table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS campaigns (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                target_segment TEXT,
                marketing_channels TEXT,
                generated_content TEXT,
                status TEXT DEFAULT 'Draft',
                budget REAL DEFAULT 0.0,
                estimated_roi REAL,
                conversion_count INTEGER
            )
        """)
        conn.commit()


# CRUD Helpers

def get_inventory_levels(db_path: str = "sme_assistant.db") -> List[Dict[str, Any]]:
    results = []
    with get_db_connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM inventory").fetchall()
        for r in rows:
            results.append({
                "sku": r["sku"],
                "product_name": r["product_name"],
                "category": r["category"],
                "stock_quantity": r["stock_quantity"],
                "price": r["price"],
                "reorder_point": r["reorder_point"],
                "days_in_stock_unsold": r["days_in_stock_unsold"]
            })
    return results


def update_stock(sku: str, qty: int, db_path: str = "sme_assistant.db") -> bool:
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE inventory SET stock_quantity = ? WHERE sku = ?",
            (qty, sku)
        )
        conn.commit()
        return cursor.rowcount > 0


def get_customers(db_path: str = "sme_assistant.db") -> List[Dict[str, Any]]:
    """
    Fetches all customers and converts database records to nested dictionaries
    matching the structure of Pydantic model CustomerProfile.
    """
    results = []
    with get_db_connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM customers").fetchall()
        for r in rows:
            purchase_history = []
            if r["purchase_history"]:
                try:
                    purchase_history = json.loads(r["purchase_history"])
                except json.JSONDecodeError:
                    purchase_history = []
            
            results.append({
                "customer_id": r["id"],
                "name": r["name"],
                "contact_info": {
                    "email": r["email"],
                    "phone": r["phone"],
                    "preferred_channel": r["preferred_channel"]
                },
                "purchase_history": purchase_history,
                "total_spend": r["total_spend"],
                "last_purchase_date": r["last_purchase_date"],
                "notes": r["notes"]
            })
    return results


def get_customer_segments(criteria: Dict[str, Any], db_path: str = "sme_assistant.db") -> List[Dict[str, Any]]:
    """
    Evaluates list of customers and filters them based on a criteria dictionary.
    Supported criteria options:
      - min_spend (float): Matches customers with total_spend >= min_spend
      - preferred_channel (str): Matches contact_info.preferred_channel
      - purchased_item (str): Matches customers who have the given item SKU in their purchase_history
      - inactive_days_min (int): Matches customers with days since last purchase >= inactive_days_min
    """
    customers = get_customers(db_path)
    filtered = []
    
    # Simple helper to calculate days since last purchase if last_purchase_date is in YYYY-MM-DD format
    from datetime import datetime
    today = datetime.now()
    
    for cust in customers:
        match = True
        
        if "min_spend" in criteria:
            if cust["total_spend"] < criteria["min_spend"]:
                match = False
                
        if "preferred_channel" in criteria:
            if cust["contact_info"].get("preferred_channel") != criteria["preferred_channel"]:
                match = False
                
        if "purchased_item" in criteria:
            purchased_skus = [item.get("item_id") for item in cust["purchase_history"]]
            if criteria["purchased_item"] not in purchased_skus:
                match = False
                
        if "inactive_days_min" in criteria:
            last_date_str = cust["last_purchase_date"]
            if not last_date_str:
                # If they never purchased, treat them as inactive
                pass
            else:
                try:
                    last_date = datetime.strptime(last_date_str, "%Y-%m-%d")
                    days_diff = (today - last_date).days
                    if days_diff < criteria["inactive_days_min"]:
                        match = False
                except ValueError:
                    # Date formatting issue
                    match = False
                    
        if match:
            filtered.append(cust)
            
    return filtered


def get_leads(db_path: str = "sme_assistant.db") -> List[Dict[str, Any]]:
    """
    Fetches all pipeline leads from the database.
    Returns a list of dictionaries matching the TrackedLead schema.
    """
    results = []
    with get_db_connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM leads").fetchall()
        for r in rows:
            results.append({
                "lead_id": r["id"],
                "source": r["source"],
                "company_name": r["company_name"],
                "contact_name": r["contact_name"],
                "estimated_deal_size": r["estimated_deal_size"],
                "lead_score": r["lead_score"],
                "status": r["status"],
                "next_action": r["next_action"]
            })
    return results


def get_campaigns(db_path: str = "sme_assistant.db") -> List[Dict[str, Any]]:
    """
    Fetches all campaigns from the database.
    Returns a list of dictionaries matching the ActiveCampaign schema.
    """
    results = []
    with get_db_connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM campaigns").fetchall()
        for r in rows:
            channels = []
            content = {}
            if r["marketing_channels"]:
                try:
                    channels = json.loads(r["marketing_channels"])
                except json.JSONDecodeError:
                    channels = []
            if r["generated_content"]:
                try:
                    content = json.loads(r["generated_content"])
                except json.JSONDecodeError:
                    content = {}
                    
            results.append({
                "campaign_id": r["id"],
                "title": r["title"],
                "target_segment": r["target_segment"],
                "marketing_channels": channels,
                "generated_content": content,
                "status": r["status"],
                "budget": r["budget"],
                "estimated_roi": r["estimated_roi"],
                "conversion_count": r["conversion_count"]
            })
    return results


def add_campaign(campaign_data: Dict[str, Any], db_path: str = "sme_assistant.db") -> str:
    """
    Inserts a new campaign record into the database.
    Accepts raw dicts, or Pydantic models with model_dump() / dict() functions.
    Returns the campaign ID.
    """
    if hasattr(campaign_data, "model_dump"):
        data = campaign_data.model_dump()
    elif hasattr(campaign_data, "dict"):
        data = campaign_data.dict()
    else:
        data = campaign_data

    campaign_id = data.get("campaign_id") or data.get("id")
    if not campaign_id:
        import uuid
        campaign_id = f"CAMP-{uuid.uuid4().hex[:6].upper()}"

    channels_json = json.dumps(data.get("marketing_channels", []))
    content_json = json.dumps(data.get("generated_content", {}))

    with get_db_connection(db_path) as conn:
        conn.execute("""
            INSERT INTO campaigns (id, title, target_segment, marketing_channels, generated_content, status, budget, estimated_roi, conversion_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            campaign_id,
            data.get("title", "Untitled Campaign"),
            data.get("target_segment", ""),
            channels_json,
            content_json,
            data.get("status", "Draft"),
            data.get("budget", 0.0),
            data.get("estimated_roi"),
            data.get("conversion_count")
        ))
        conn.commit()
    return campaign_id


# --- Demo Seeding ---

def seed_db(db_path: str = "sme_assistant.db") -> None:
    """
    Seeds the database with high-quality, rich synthetic records for demonstrations.
    This wipes existing database records in these tables first.
    """
    init_db(db_path)
    
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        
        # Clear existing data
        cursor.execute("DELETE FROM customers")
        cursor.execute("DELETE FROM inventory")
        cursor.execute("DELETE FROM leads")
        cursor.execute("DELETE FROM campaigns")
        
        # --- Seed Inventory (8 Items) ---
        inventory_items = [
            ("SKU-KT-BLU", "Cotton Kurta - Blue", "Apparel", 45, 1200.0, 10, 125),   # Slow-moving
            ("SKU-SD-BRN", "Leather Sandals - Brown", "Footwear", 30, 1800.0, 5, 95), # Slow-moving
            ("SKU-SR-RED", "Silk Saree - Red", "Apparel", 2, 4500.0, 5, 5),          # Low stock
            ("SKU-SH-WHT", "Linen Shirt - White", "Apparel", 25, 1500.0, 8, 12),
            ("SKU-DP-PNK", "Designer Dupatta - Pink", "Accessories", 15, 600.0, 5, 18),
            ("SKU-KT-YEL", "Chanderi Kurti - Yellow", "Apparel", 40, 2200.0, 10, 80),
            ("SKU-ER-JHM", "Silver Earrings - Jhumka", "Accessories", 8, 950.0, 10, 4), # Low stock
            ("SKU-BG-TOT", "Handcrafted Tote Bag", "Accessories", 12, 1400.0, 4, 25)
        ]
        cursor.executemany("""
            INSERT INTO inventory (sku, product_name, category, stock_quantity, price, reorder_point, days_in_stock_unsold)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, inventory_items)
        
        # --- Seed Customers (18 Profiles) ---
        customers = [
            (
                "CUST001", "Kavya Sharma", "kavya.sharma@example.com", "+919876543210", "whatsapp",
                json.dumps([
                    {"item_id": "SKU-SR-RED", "timestamp": "2026-05-10T14:30:00", "amount": 4500.0},
                    {"item_id": "SKU-DP-PNK", "timestamp": "2026-06-01T11:00:00", "amount": 600.0}
                ]),
                5100.0, "2026-06-01", "VIP customer who loves silk apparel and designer dupattas. Prefers WhatsApp."
            ),
            (
                "CUST002", "Aarav Mehta", "aarav.mehta@example.com", "+919876543211", "email",
                json.dumps([
                    {"item_id": "SKU-SH-WHT", "timestamp": "2026-04-15T10:15:00", "amount": 1500.0}
                ]),
                1500.0, "2026-04-15", "Enjoys minimal linen styles. Checks email weekly."
            ),
            (
                "CUST003", "Ananya Sen", "ananya.sen@example.com", "+919876543212", "whatsapp",
                json.dumps([]),
                0.0, None, "New sign-up. Showed high interest in accessories like Jhumkas during welcome conversation."
            ),
            (
                "CUST004", "Rohan Das", "rohan.das@example.com", "+919876543213", "sms",
                json.dumps([
                    {"item_id": "SKU-SD-BRN", "timestamp": "2025-12-25T16:00:00", "amount": 1800.0}
                ]),
                1800.0, "2025-12-25", "Inactive customer. Purchased leather sandals last winter. No touchpoint since."
            ),
            (
                "CUST005", "Priya Patel", "priya.patel@example.com", "+919876543214", "whatsapp",
                json.dumps([
                    {"item_id": "SKU-KT-BLU", "timestamp": "2026-01-10T12:00:00", "amount": 1200.0},
                    {"item_id": "SKU-KT-YEL", "timestamp": "2026-02-14T15:30:00", "amount": 2200.0}
                ]),
                3400.0, "2026-02-14", "Likes cotton kurtas. Has been inactive for over 4 months."
            ),
            (
                "CUST006", "Vikram Singh", "vikram.singh@example.com", "+919876543215", "email",
                json.dumps([
                    {"item_id": "SKU-SH-WHT", "timestamp": "2026-06-20T10:00:00", "amount": 1500.0},
                    {"item_id": "SKU-SH-WHT", "timestamp": "2026-06-25T11:45:00", "amount": 1500.0}
                ]),
                3000.0, "2026-06-25", "Frequent shopper. Very responsive to linen product additions."
            ),
            (
                "CUST007", "Meera Nair", "meera.nair@example.com", "+919876543216", "whatsapp",
                json.dumps([
                    {"item_id": "SKU-SR-RED", "timestamp": "2026-06-12T17:10:00", "amount": 4500.0},
                    {"item_id": "SKU-BG-TOT", "timestamp": "2026-06-18T14:20:00", "amount": 1400.0},
                    {"item_id": "SKU-ER-JHM", "timestamp": "2026-06-22T19:30:00", "amount": 950.0}
                ]),
                6850.0, "2026-06-22", "High-value VIP customer. Interested in boutique designer sarees and jewelry."
            ),
            (
                "CUST008", "Kabir Kapoor", "kabir.kapoor@example.com", "+919876543217", "sms",
                json.dumps([]),
                0.0, None, "Subscribed to SMS alerts. Never purchased. Registered from social media."
            ),
            (
                "CUST009", "Divya Reddy", "divya.reddy@example.com", "+919876543218", "whatsapp",
                json.dumps([
                    {"item_id": "SKU-DP-PNK", "timestamp": "2026-03-05T13:00:00", "amount": 600.0}
                ]),
                600.0, "2026-03-05", "Purchased a single accessory. Good target for budget-friendly recommendations."
            ),
            (
                "CUST010", "Aditya Joshi", "aditya.joshi@example.com", "+919876543219", "email",
                json.dumps([
                    {"item_id": "SKU-SD-BRN", "timestamp": "2026-05-20T10:30:00", "amount": 1800.0}
                ]),
                1800.0, "2026-05-20", "Prefers traditional footwear. Enjoys curated apparel catalogs."
            ),
            (
                "CUST011", "Rhea Bose", "rhea.bose@example.com", "+919876543220", "whatsapp",
                json.dumps([
                    {"item_id": "SKU-KT-BLU", "timestamp": "2026-02-10T14:00:00", "amount": 1200.0}
                ]),
                1200.0, "2026-02-10", "Purchased the Blue Cotton Kurta. Has not interacted since February."
            ),
            (
                "CUST012", "Ishaan Malhotra", "ishaan.m@example.com", "+919876543221", "email",
                json.dumps([
                    {"item_id": "SKU-SH-WHT", "timestamp": "2026-01-15T09:00:00", "amount": 1500.0}
                ]),
                1500.0, "2026-01-15", "Inactive linen buyer. Good recipient for clearance sales."
            ),
            (
                "CUST013", "Sanya Gupta", "sanya.g@example.com", "+919876543222", "whatsapp",
                json.dumps([
                    {"item_id": "SKU-BG-TOT", "timestamp": "2026-06-24T16:20:00", "amount": 1400.0}
                ]),
                1400.0, "2026-06-24", "Recent buyer. Inquired about handmade bags. Active responder."
            ),
            (
                "CUST014", "Neha Verma", "neha.verma@example.com", "+919876543223", "whatsapp",
                json.dumps([]),
                0.0, None, "Registered through WhatsApp chatbot. Interested in festive sales."
            ),
            (
                "CUST015", "Rahul Bhatia", "rahul.b@example.com", "+919876543224", "sms",
                json.dumps([
                    {"item_id": "SKU-SD-BRN", "timestamp": "2025-11-12T11:00:00", "amount": 1800.0}
                ]),
                1800.0, "2025-11-12", "Long term inactive. Re-engagement candidate via SMS."
            ),
            (
                "CUST016", "Tara Deshmukh", "tara.d@example.com", "+919876543225", "email",
                json.dumps([
                    {"item_id": "SKU-SR-RED", "timestamp": "2026-06-15T15:00:00", "amount": 4500.0}
                ]),
                4500.0, "2026-06-15", "VIP candidate. Purchased Silk Saree. Responds to styling tips."
            ),
            (
                "CUST017", "Zain Khan", "zain.khan@example.com", "+919876543226", "whatsapp",
                json.dumps([
                    {"item_id": "SKU-KT-YEL", "timestamp": "2026-05-30T10:00:00", "amount": 2200.0}
                ]),
                2200.0, "2026-05-30", "Buys premium kurtis. Very responsive to whatsapp catalogs."
            ),
            (
                "CUST018", "Pooja Pillai", "pooja.p@example.com", "+919876543227", "whatsapp",
                json.dumps([
                    {"item_id": "SKU-DP-PNK", "timestamp": "2026-06-10T12:00:00", "amount": 600.0},
                    {"item_id": "SKU-ER-JHM", "timestamp": "2026-06-11T13:00:00", "amount": 950.0}
                ]),
                1550.0, "2026-06-11", "Accessory enthusiast. Often matches pink dupattas with silver earrings."
            )
        ]
        cursor.executemany("""
            INSERT INTO customers (id, name, email, phone, preferred_channel, purchase_history, total_spend, last_purchase_date, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, customers)
        
        # --- Seed Leads (6 Pipeline Leads) ---
        leads = [
            ("LEAD001", "Web Form", "Boutique Elegance", "Rahul Verma", 15000.0, 8.5, "New", "Send wholesale apparel pricing catalog"),
            ("LEAD002", "Cold Search", "IndoCrafts Retail", "Priya Nair", 35000.0, 6.2, "Contacted", "Follow up on phone call regarding bulk saree orders"),
            ("LEAD003", "Instagram DM", "Styles of India", "Amit Saxena", 8000.0, 7.8, "Nurturing", "Draft customized brochure for boutique kurtas"),
            ("LEAD004", "Referral", "Royal Heritage Store", "Suresh Gupta", 50000.0, 9.4, "Qualified", "Draft proposal for exclusive retail partnership"),
            ("LEAD005", "Web Form", "Ethnic Wear House", "Shweta Rao", 12000.0, 5.0, "New", "Verify email address and send welcome message"),
            ("LEAD006", "Cold Search", "Desi Trendz", "Karan Johar", 20000.0, 4.1, "Contacted", "Send follow-up email about customized handcraft collection")
        ]
        cursor.executemany("""
            INSERT INTO leads (id, source, company_name, contact_name, estimated_deal_size, lead_score, status, next_action)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, leads)
        
        # --- Seed Campaigns (2 Campaigns) ---
        campaigns = [
            (
                "CAMP001", "Festive Silk Clearance", "Premium VIP Customers",
                json.dumps(["whatsapp", "email"]),
                json.dumps({
                    "whatsapp": "Hello {name}, enjoy an exclusive 20% off on our handloomed Silk Saree collection! Visit us today.",
                    "email": "Subject: Exclusive VIP Offer: 20% Off Silk Sarees\n\nDear {name},\n\nWe are delighted to offer you 20% off on our collection..."
                }),
                "Sent", 800.0, 3.2, 14
            ),
            (
                "CAMP002", "Slow-Moving Apparel Push", "Cotton Lovers Segment",
                json.dumps(["whatsapp"]),
                json.dumps({
                    "whatsapp": "Hi {name}! Check out our classic Blue Cotton Kurta, perfect for daily comfort. Special offer: 15% off today only!"
                }),
                "Draft", 300.0, 2.5, 0
            )
        ]
        cursor.executemany("""
            INSERT INTO campaigns (id, title, target_segment, marketing_channels, generated_content, status, budget, estimated_roi, conversion_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, campaigns)
        
        conn.commit()


if __name__ == "__main__":
    db_file = "sme_assistant.db"
    print(f"Initializing and seeding local SQLite database: {db_file}...")
    seed_db(db_file)
    print("Database seeding completed successfully!")
    
    # Simple validation stats
    with get_db_connection(db_file) as conn:
        print(f"Total Customers: {conn.execute('SELECT count(*) FROM customers').fetchone()[0]}")
        print(f"Total Inventory SKU: {conn.execute('SELECT count(*) FROM inventory').fetchone()[0]}")
        print(f"Total Pipeline Leads: {conn.execute('SELECT count(*) FROM leads').fetchone()[0]}")
        print(f"Total Campaigns: {conn.execute('SELECT count(*) FROM campaigns').fetchone()[0]}")
        
        # Low stock query
        print("\nLow stock alerts (stock < reorder_point):")
        for row in conn.execute("SELECT product_name, stock_quantity, reorder_point FROM inventory WHERE stock_quantity < reorder_point"):
            print(f"  - {row['product_name']}: stock={row['stock_quantity']} (reorder={row['reorder_point']})")
            
        # Slow moving inventory query
        print("\nSlow moving items (unsold > 90 days):")
        for row in conn.execute("SELECT product_name, days_in_stock_unsold FROM inventory WHERE days_in_stock_unsold > 90"):
            print(f"  - {row['product_name']}: {row['days_in_stock_unsold']} days unsold")
