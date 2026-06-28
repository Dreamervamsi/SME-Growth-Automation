# SME-Growth-Automation

An autonomous multi-agent system to automate daily operations for small businesses, orchestrating specialized agents for lead generation, inventory tracking, CRM management, and marketing automation.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
TELEGRAM_BOT_TOKEN=your_telegram_bot_token   # optional; mock mode if omitted
```

## Run the dashboard

```bash
streamlit run app.py
```

## Run tests

```bash
pytest tests/ -v
```

## Seed the database manually

```bash
python -m src.langraph.database
```

## Architecture

- **Orchestrator** — routes user requests to specialist agents or finishes the workflow
- **CRM Agent** — customer segmentation and profile analysis
- **Stock Agent** — low-stock and slow-moving inventory flags
- **Leads Agent** — lead scoring and outreach planning
- **Marketing Agent** — campaign copy generation (with human-in-the-loop approval in the UI)
- **Responder** — delivers the final summary to the user

Data is stored in SQLite (`sme_assistant.db`). Utility helpers in `tools.py` support inventory import/export and Telegram notifications.

The Streamlit chat UI streams specialist and final responses token-by-token via LangGraph custom events, so users see progress immediately instead of waiting for the full workflow to finish.
