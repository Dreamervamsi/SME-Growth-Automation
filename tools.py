import os
import json
import sqlite3
import pandas as pd
import requests
from typing import List, Dict, Any, Union

# Import database connection helper from our package
from src.langraph.database import get_db_connection

def import_inventory_from_excel(file_path: str, db_path: str = "sme_assistant.db") -> Dict[str, Any]:
    """
    Reads an uploaded spreadsheet (CSV or Excel) and bulk-inserts or replaces data
    in the SQLite 'inventory' table. This allows users to onboard data instantly.
    
    Args:
        file_path (str): Path to the CSV or Excel file.
        db_path (str): Path to the SQLite database file.
        
    Returns:
        dict: A status dictionary containing success/error status and execution details.
    """
    try:
        # Determine file type and read into pandas DataFrame
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.csv':
            df = pd.read_csv(file_path)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            return {
                "status": "error",
                "message": f"Unsupported file format '{ext}'. Only CSV and Excel (.xlsx, .xls) are supported."
            }

        # Normalize column names (lowercase, replace spaces/hyphens with underscores)
        normalized_cols = {col: col.strip().lower().replace(" ", "_").replace("-", "_") for col in df.columns}
        df = df.rename(columns=normalized_cols)

        # Expected DB columns: sku, product_name, category, stock_quantity, price, reorder_point, days_in_stock_unsold
        # Map common variations to the expected column names
        mapping_variations = {
            "product_name": ["product_name", "product", "name", "item_name", "title"],
            "stock_quantity": ["stock_quantity", "quantity", "qty", "stock", "in_stock", "units"],
            "reorder_point": ["reorder_point", "reorder", "reorder_level", "threshold", "min_stock"],
            "days_in_stock_unsold": ["days_in_stock_unsold", "days_unsold", "unsold_days", "days_in_stock"]
        }

        for target_col, variations in mapping_variations.items():
            if target_col not in df.columns:
                for var in variations:
                    if var in df.columns:
                        df = df.rename(columns={var: target_col})
                        break

        # Validate required columns
        required_cols = ["sku", "product_name", "stock_quantity", "price", "reorder_point"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return {
                "status": "error",
                "message": f"Spreadsheet is missing required columns: {', '.join(missing_cols)}. Found columns: {', '.join(df.columns)}"
            }

        # Ensure sensible defaults for optional columns
        if "category" not in df.columns:
            df["category"] = "General"
        else:
            df["category"] = df["category"].fillna("General")

        if "days_in_stock_unsold" not in df.columns:
            df["days_in_stock_unsold"] = 0
        else:
            df["days_in_stock_unsold"] = df["days_in_stock_unsold"].fillna(0).astype(int)

        # Convert types to match SQLite schemas
        df["sku"] = df["sku"].astype(str).str.strip()
        df["product_name"] = df["product_name"].astype(str).str.strip()
        df["stock_quantity"] = df["stock_quantity"].fillna(0).astype(int)
        df["price"] = df["price"].fillna(0.0).astype(float)
        df["reorder_point"] = df["reorder_point"].fillna(0).astype(int)

        # Bulk insert or replace in SQLite
        imported_count = 0
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            for _, row in df.iterrows():
                cursor.execute("""
                    INSERT OR REPLACE INTO inventory (sku, product_name, category, stock_quantity, price, reorder_point, days_in_stock_unsold)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    row["sku"],
                    row["product_name"],
                    row["category"],
                    row["stock_quantity"],
                    row["price"],
                    row["reorder_point"],
                    row["days_in_stock_unsold"]
                ))
                imported_count += 1
            conn.commit()

        return {
            "status": "success",
            "message": f"Successfully imported/updated {imported_count} items in the inventory database.",
            "count": imported_count
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"An error occurred while processing the spreadsheet: {str(e)}"
        }


def export_reorder_list_to_csv(low_stock_items: List[Any], file_path: str = "reorder_list.csv") -> Dict[str, Any]:
    """
    Takes flagged low-stock items (dict format, sqlite3.Row format, or Pydantic models) 
    and writes them into a clean, downloadable CSV file for the owner.
    
    Args:
        low_stock_items (list): List of low-stock inventory items.
        file_path (str): The filename/path to save the CSV export to.
        
    Returns:
        dict: A status dictionary containing the status, message, and path of the file.
    """
    try:
        if not low_stock_items:
            # Create an empty CSV file with appropriate headers to avoid errors
            df = pd.DataFrame(columns=["sku", "product_name", "category", "stock_quantity", "price", "reorder_point", "suggested_reorder_qty"])
            df.to_csv(file_path, index=False)
            return {
                "status": "success",
                "message": "No low-stock items provided. Generated empty reorder template.",
                "file_path": file_path,
                "count": 0
            }

        # Normalize items list to standard dictionaries
        records = []
        for item in low_stock_items:
            if hasattr(item, "model_dump"):
                record = item.model_dump()
            elif hasattr(item, "dict"):
                record = item.dict()
            elif hasattr(item, "keys"):
                record = dict(item)
            elif isinstance(item, dict):
                record = item
            else:
                # Fallback attributes conversion
                record = {
                    "sku": getattr(item, "sku", ""),
                    "product_name": getattr(item, "product_name", ""),
                    "category": getattr(item, "category", ""),
                    "stock_quantity": getattr(item, "stock_quantity", 0),
                    "price": getattr(item, "price", 0.0),
                    "reorder_point": getattr(item, "reorder_point", 0),
                    "days_in_stock_unsold": getattr(item, "days_in_stock_unsold", 0)
                }
            
            # Calculate a helper suggested reorder quantity: (reorder_point * 2) - stock_quantity
            # (or a default restock calculation)
            stock_qty = record.get("stock_quantity", 0)
            reorder_pt = record.get("reorder_point", 0)
            suggested_reorder = max(0, (reorder_pt * 2) - stock_qty)
            record["suggested_reorder_qty"] = suggested_reorder
            
            records.append(record)

        df = pd.DataFrame(records)
        df.to_csv(file_path, index=False)

        return {
            "status": "success",
            "message": f"Successfully exported {len(records)} low-stock items to CSV.",
            "file_path": file_path,
            "count": len(records)
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to export reorder list to CSV: {str(e)}"
        }


def send_whatsapp_or_telegram_notification(recipient_id: str, message_text: str) -> Dict[str, Any]:
    """
    Sends an automated notification to a Telegram channel/chat using the Telegram Bot API.
    If no TELEGRAM_BOT_TOKEN environment variable is defined or if it matches a dummy key, 
    the system falls back to a simulated mock response to ensure robust demonstration.
    
    Args:
        recipient_id (str): The Telegram chat ID of the user/owner.
        message_text (str): The text message to send.
        
    Returns:
        dict: A status dictionary containing the dispatch status and logs.
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    
    # Check for empty or placeholder tokens
    if not token or token.strip() in ["", "dummy_key_to_allow_import", "YOUR_TELEGRAM_BOT_TOKEN"]:
        print(f"[MOCK TELEGRAM NOTIFICATION] To: {recipient_id}\nMessage: {message_text}\n")
        return {
            "status": "success",
            "message": "[MOCK] Notification printed to console/logs (no real TELEGRAM_BOT_TOKEN is set).",
            "recipient_id": recipient_id,
            "body": message_text
        }

    url = f"https://api.telegram.org/bot{token.strip()}/sendMessage"
    payload = {
        "chat_id": recipient_id,
        "text": message_text,
        "parse_mode": "HTML"
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        resp_data = response.json()
        
        if response.status_code == 200 and resp_data.get("ok"):
            return {
                "status": "success",
                "message": "Notification successfully sent via Telegram Bot.",
                "recipient_id": recipient_id,
                "message_id": resp_data.get("result", {}).get("message_id")
            }
        else:
            return {
                "status": "error",
                "message": f"Telegram API returned an error: {resp_data.get('description', 'Unknown error')}",
                "recipient_id": recipient_id
            }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to connect to Telegram API: {str(e)}",
            "recipient_id": recipient_id
        }
