"""
Example 2: LangChain Integration

This example demonstrates how to:
- Extract metadata from LangChain agents automatically
- Use the callback handler for runtime logging
- Attach AgentFacts to an existing agent workflow

NOTE: This example requires langchain and langchain-openai to be installed:
    pip install agentfacts[langchain] langchain-openai
"""

import os
from typing import Optional

# Check if LangChain is available
try:
    from langchain_core.tools import tool
    from langchain_core.prompts import ChatPromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("LangChain not installed. Install with: pip install agentfacts[langchain]")

from agentfacts import AgentFacts
from agentfacts.models import BaselineModel, Capability, ModelProvider


def create_mock_agent():
    """Create a mock agent-like object for demonstration without API keys."""

    class MockLLM:
        """Mock LLM that simulates LangChain LLM interface."""
        model_name = "gpt-4"
        temperature = 0.7
        max_tokens = 2048

    class MockTool:
        """Mock tool that simulates LangChain tool interface."""
        def __init__(self, name: str, description: str):
            self.name = name
            self.description = description
            self.args_schema = None

    class MockAgent:
        """Mock agent that simulates LangChain agent interface."""
        def __init__(self):
            self.llm = MockLLM()

    class MockAgentExecutor:
        """Mock AgentExecutor that simulates LangChain AgentExecutor."""
        def __init__(self):
            self.agent = MockAgent()
            self.tools = [
                MockTool("calculator", "Perform mathematical calculations"),
                MockTool("web_search", "Search the web for information"),
                MockTool("file_reader", "Read contents of files"),
            ]
            self.max_iterations = 10
            self.max_execution_time = 60

    return MockAgentExecutor()


def example_with_mock_agent():
    """Demonstrate AgentFacts with a mock agent (no API key needed)."""
    print("\n" + "-" * 50)
    print("Using Mock Agent (no API key required)")
    print("-" * 50)

    # Create mock agent
    mock_executor = create_mock_agent()

    # Extract metadata using AgentFacts
    facts = AgentFacts.from_langchain(
        mock_executor,
        name="Research Assistant",
        description="A helpful research agent powered by GPT-4",
    )

    print(f"\nExtracted Metadata:")
    print(f"  Name:        {facts.name}")
    print(f"  DID:         {facts.did}")
    print(f"  Model:       {facts.metadata.agent.model.name}")
    print(f"  Provider:    {facts.metadata.agent.model.provider.value}")
    print(f"  Temperature: {facts.metadata.agent.model.temperature}")

    print(f"\nCapabilities ({len(facts.metadata.agent.capabilities)}):")
    for cap in facts.metadata.agent.capabilities:
        print(f"  - {cap.name}: {cap.description}")
        print(f"    Risk Level: {cap.risk_level}")

    print(f"\nConstraints:")
    print(f"  Max Iterations: {facts.metadata.policy.constraints.max_iterations}")
    print(f"  Timeout:        {facts.metadata.policy.constraints.timeout_seconds}s")

    # Sign the metadata
    facts.sign()
    print(f"\nMetadata signed: {facts.is_signed}")

    # Alternative: Use the convenience factory (one-liner)
    print("\n  Alternative: Using from_langchain_signed()...")
    signed_facts = AgentFacts.from_langchain_signed(
        mock_executor,
        name="Research Assistant (Signed)",
    )
    print(f"  One-liner signed: {signed_facts.is_signed}")

    return facts


def example_with_real_langchain():
    """Demonstrate AgentFacts with a real LangChain agent."""
    if not LANGCHAIN_AVAILABLE:
        print("\nSkipping real LangChain example (not installed)")
        return None

    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("\nSkipping real LangChain example (OPENAI_API_KEY not set)")
        return None

    print("\n" + "-" * 50)
    print("Using Real LangChain Agent")
    print("-" * 50)

    from langchain_openai import ChatOpenAI
    from langchain.agents import AgentExecutor, create_tool_calling_agent

    # Define tools
    @tool
    def calculator(expression: str) -> str:
        """Evaluate a mathematical expression."""
        try:
            return str(eval(expression))
        except Exception as e:
            return f"Error: {e}"

    @tool
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        # Mock implementation
        return f"The weather in {city} is sunny and 72°F"

    # Create LLM
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)

    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # Create agent
    tools = [calculator, get_weather]
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Extract metadata
    facts = AgentFacts.from_langchain(
        executor,
        name="Weather & Math Agent",
        description="An agent that can check weather and do math",
    )

    facts.sign()
    return facts


def example_callback_handler():
    """Demonstrate the callback handler for runtime logging."""
    print("\n" + "-" * 50)
    print("Using Callback Handler for Runtime Logging")
    print("-" * 50)

    from agentfacts.integrations.langchain.callback import AgentFactsCallbackHandler

    # Create AgentFacts
    facts = AgentFacts(
        name="Logged Agent",
        baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
    )

    # Create callback handler
    handler = AgentFactsCallbackHandler(
        agent_facts=facts,
        log_llm_calls=True,
        log_tool_calls=True,
        log_errors=True,
    )

    # Simulate some events (in real usage, LangChain calls these automatically)
    from uuid import uuid4

    # Simulate chain start
    handler.on_chain_start(
        serialized={"name": "TestChain"},
        inputs={"query": "test"},
        run_id=uuid4(),
    )

    # Simulate LLM call
    handler.on_llm_start(
        serialized={"name": "gpt-4"},
        prompts=["Hello, how are you?"],
        run_id=uuid4(),
    )

    # Simulate tool call
    handler.on_tool_start(
        serialized={"name": "calculator"},
        input_str="2 + 2",
        run_id=uuid4(),
    )

    handler.on_tool_end(output="4", run_id=uuid4())

    # Get summary
    summary = handler.get_summary()
    print(f"\nCallback Summary:")
    print(f"  Total Events:      {summary['total_events']}")
    print(f"  LLM Invocations:   {summary['llm_invocations']}")
    print(f"  Tool Invocations:  {summary['tool_invocations']}")
    print(f"  Tools Used:        {summary['tools_used']}")

    # Get all events
    print(f"\nRecorded Events:")
    for event in handler.get_events():
        print(f"  - {event['type']}: {event.get('tool_name', event.get('model', ''))}")


def main():
    print("=" * 60)
    print("AgentFacts SDK - LangChain Integration Example")
    print("=" * 60)

    # Example 1: Mock agent (always works)
    facts = example_with_mock_agent()

    # Example 2: Real LangChain (requires API key)
    example_with_real_langchain()

    # Example 3: Callback handler
    example_callback_handler()

    # Show final JSON output
    print("\n" + "-" * 50)
    print("Final Agent Metadata (JSON)")
    print("-" * 50)
    print(facts.to_json())

    print("\n" + "=" * 60)
    print("LangChain integration example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
