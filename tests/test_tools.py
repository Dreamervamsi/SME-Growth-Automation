import os
import sys

# Ensure root directory is in sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pytest
import pandas as pd
import sqlite3
from src.langraph import init_db, get_inventory_levels
from tools import (
    import_inventory_from_excel,
    export_reorder_list_to_csv,
    send_whatsapp_or_telegram_notification
)

TEST_DB_PATH = "test_tools_sme_assistant.db"
TEMP_CSV_PATH = "test_inventory_import.csv"
TEMP_EXPORT_PATH = "test_reorder_export.csv"

@pytest.fixture(autouse=True)
def setup_and_teardown():
    # Clean up before tests
    for path in [TEST_DB_PATH, TEMP_CSV_PATH, TEMP_EXPORT_PATH]:
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
    
    # Initialize the test database
    init_db(TEST_DB_PATH)
    
    yield
    
    # Clean up after tests
    for path in [TEST_DB_PATH, TEMP_CSV_PATH, TEMP_EXPORT_PATH]:
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


def test_import_inventory_from_excel_csv():
    # 1. Prepare dummy CSV data
    data = {
        "sku": ["SKU-TST-001", "SKU-TST-002"],
        "product_name": ["Test Widget A", "Test Widget B"],
        "category": ["Widgets", "Gadgets"],
        "stock_quantity": [10, 5],
        "price": [19.99, 99.50],
        "reorder_point": [20, 10],
        "days_in_stock_unsold": [12, 145]
    }
    df = pd.DataFrame(data)
    df.to_csv(TEMP_CSV_PATH, index=False)

    # 2. Call import
    result = import_inventory_from_excel(TEMP_CSV_PATH, db_path=TEST_DB_PATH)
    assert result["status"] == "success"
    assert result["count"] == 2

    # 3. Verify database contents
    inventory = get_inventory_levels(TEST_DB_PATH)
    inventory_by_sku = {item["sku"]: item for item in inventory}
    
    assert "SKU-TST-001" in inventory_by_sku
    assert inventory_by_sku["SKU-TST-001"]["product_name"] == "Test Widget A"
    assert inventory_by_sku["SKU-TST-001"]["stock_quantity"] == 10
    assert inventory_by_sku["SKU-TST-001"]["price"] == 19.99
    assert inventory_by_sku["SKU-TST-001"]["reorder_point"] == 20
    assert inventory_by_sku["SKU-TST-001"]["days_in_stock_unsold"] == 12

    assert "SKU-TST-002" in inventory_by_sku
    assert inventory_by_sku["SKU-TST-002"]["days_in_stock_unsold"] == 145


def test_import_inventory_missing_columns():
    # Prepare dummy CSV data with missing required column (e.g. price)
    data = {
        "sku": ["SKU-ERR-001"],
        "product_name": ["Error Widget"],
        "stock_quantity": [10],
        "reorder_point": [20]
    }
    df = pd.DataFrame(data)
    df.to_csv(TEMP_CSV_PATH, index=False)

    result = import_inventory_from_excel(TEMP_CSV_PATH, db_path=TEST_DB_PATH)
    assert result["status"] == "error"
    assert "missing required columns" in result["message"]


def test_export_reorder_list_to_csv():
    # 1. Mock low-stock items (mix of dicts and Pydantic-like / object properties)
    class DummyItem:
        def __init__(self, sku, name, category, stock, price, reorder):
            self.sku = sku
            self.product_name = name
            self.category = category
            self.stock_quantity = stock
            self.price = price
            self.reorder_point = reorder

    items = [
        {"sku": "SKU-LOW-001", "product_name": "Low Stock A", "category": "A", "stock_quantity": 2, "price": 10.0, "reorder_point": 5},
        DummyItem("SKU-LOW-002", "Low Stock B", "B", 1, 15.0, 10)
    ]

    # 2. Call export
    result = export_reorder_list_to_csv(items, TEMP_EXPORT_PATH)
    assert result["status"] == "success"
    assert result["count"] == 2
    assert os.path.exists(TEMP_EXPORT_PATH)

    # 3. Read back CSV and verify suggested_reorder_qty
    df = pd.read_csv(TEMP_EXPORT_PATH)
    assert len(df) == 2
    assert df.loc[df["sku"] == "SKU-LOW-001", "suggested_reorder_qty"].values[0] == 8  # (5 * 2) - 2 = 8
    assert df.loc[df["sku"] == "SKU-LOW-002", "suggested_reorder_qty"].values[0] == 19 # (10 * 2) - 1 = 19


def test_send_telegram_notification_mock():
    # Test with dummy token / environment
    if "TELEGRAM_BOT_TOKEN" in os.environ:
        old_token = os.environ["TELEGRAM_BOT_TOKEN"]
        del os.environ["TELEGRAM_BOT_TOKEN"]
    else:
        old_token = None

    try:
        # Without any token (or dummy)
        result = send_whatsapp_or_telegram_notification("123456", "Hello Demo")
        assert result["status"] == "success"
        assert "[MOCK]" in result["message"]
        assert result["recipient_id"] == "123456"
        assert result["body"] == "Hello Demo"
    finally:
        if old_token:
            os.environ["TELEGRAM_BOT_TOKEN"] = old_token
