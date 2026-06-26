from typing import Any, List, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, HumanMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from pydantic import Field

from src.langraph import (
    CustomerProfile,
    InventoryItem,
    TrackedLead,
    SMEState,
    create_agent_node,
    CRM_AGENT_PROMPT,
    STOCK_AGENT_PROMPT,
    LEADS_AGENT_PROMPT,
    MARKETING_AGENT_PROMPT,
)

class MockChatModel(BaseChatModel):
    responses: List[str] = Field(default_factory=list)
    calls: List[List[BaseMessage]] = Field(default_factory=list)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(messages)
        content = self.responses.pop(0) if self.responses else "Mock reply"
        message = AIMessage(content=content)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "mock-chat-model"


def test_create_agent_node_execution():
    # Instantiate mock chat model
    llm = MockChatModel(responses=["CRM response content"])

    # Create the agent node function
    crm_node = create_agent_node(llm, CRM_AGENT_PROMPT)

    # Prepare input state
    state: SMEState = {
        "messages": [HumanMessage(content="Hello CRM agent")],
        "customer_profiles": [
            CustomerProfile(
                customer_id="C001",
                name="John Doe",
                contact_info={"email": "john@example.com"},
                purchase_history=[],
                total_spend=0.0,
            )
        ],
        "current_inventory": [
            InventoryItem(
                sku="SKU1",
                product_name="Product 1",
                category="General",
                stock_quantity=10,
                price=5.0,
                reorder_point=2,
                days_in_stock_unsold=5,
            )
        ],
        "tracked_leads": [
            TrackedLead(
                lead_id="L001",
                source="Web",
                estimated_deal_size=1000.0,
                lead_score=7.5,
                status="New",
            )
        ],
        "active_campaigns": [],
        "next_agents": ["crm_agent"],
    }

    # Execute the node
    result = crm_node(state)

    # Assertions
    assert "messages" in result
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert result["messages"][0].content == "CRM response content"
    assert result["next_agents"] == ["orchestrator"]

    # Verify input messages passed to LLM
    assert len(llm.calls) == 1
    passed_messages = llm.calls[0]
    
    # First message should be the SystemMessage with the system prompt and the serialized state snapshot
    assert isinstance(passed_messages[0], SystemMessage)
    assert CRM_AGENT_PROMPT in passed_messages[0].content
    assert "C001" in passed_messages[0].content
    assert "SKU1" in passed_messages[0].content
    assert "L001" in passed_messages[0].content

    # Second message should be the original user/human message
    assert isinstance(passed_messages[1], HumanMessage)
    assert passed_messages[1].content == "Hello CRM agent"


def test_agent_prompts():
    assert "CRM (Customer Relationship Management) Agent" in CRM_AGENT_PROMPT
    assert "Stock and Inventory Agent" in STOCK_AGENT_PROMPT
    assert "Leads Management Agent" in LEADS_AGENT_PROMPT
    assert "Marketing Campaign Agent" in MARKETING_AGENT_PROMPT
