"""
Test AgentFacts introspection against various LangChain agent patterns found on GitHub.

This module tests AgentFacts against 25 different agent configurations
representing real-world patterns from popular GitHub repositories.
"""

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from agentfacts import AgentFacts, KeyPair
from agentfacts.integrations.langchain.introspector import (
    introspect_any,
    introspect_langgraph,
    introspect_llm,
    introspect_tools,
)
from agentfacts.models import ModelProvider

# =============================================================================
# Mock LangChain Components
# =============================================================================


class MockLLM:
    """Mock LLM for testing introspection."""

    def __init__(
        self,
        model_name: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        for k, v in kwargs.items():
            setattr(self, k, v)

    def invoke(self, prompt: str) -> str:
        return "Mock response"

    def generate(self, prompts: list[str]) -> Any:
        return MagicMock()


class MockChatOpenAI(MockLLM):
    """Mock ChatOpenAI for testing."""

    pass


class MockChatAnthropic(MockLLM):
    """Mock ChatAnthropic for testing."""

    pass


class MockChatGoogleGenerativeAI(MockLLM):
    """Mock Google Gemini for testing."""

    pass


class MockOllama(MockLLM):
    """Mock Ollama for local models."""

    pass


class MockTool:
    """Mock LangChain tool."""

    def __init__(
        self,
        name: str,
        description: str,
        args_schema: Any = None,
    ):
        self.name = name
        self.description = description
        self.args_schema = args_schema

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        return "Tool result"


class MockAgentExecutor:
    """Mock AgentExecutor."""

    def __init__(
        self,
        agent: Any,
        tools: list[MockTool],
        max_iterations: int = 15,
        max_execution_time: float | None = None,
        return_intermediate_steps: bool = False,
    ):
        self.agent = agent
        self.tools = tools
        self.max_iterations = max_iterations
        self.max_execution_time = max_execution_time
        self.return_intermediate_steps = return_intermediate_steps


class MockAgent:
    """Mock inner agent with LLM chain."""

    def __init__(self, llm: MockLLM):
        self.llm_chain = MagicMock()
        self.llm_chain.llm = llm
        self.llm = llm


class MockChain:
    """Mock LangChain chain."""

    def __init__(self, llm: MockLLM, tools: list[MockTool] | None = None):
        self.llm = llm
        self.tools = tools or []


class MockRunnable:
    """Mock LCEL Runnable."""

    def __init__(
        self, first: Any = None, middle: list[Any] | None = None, last: Any = None
    ):
        self.first = first
        self.middle = middle or []
        self.last = last

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        return "Runnable result"

    def batch(self, *args: Any, **kwargs: Any) -> list[Any]:
        return ["Batch result"]


class MockRunnableBinding:
    """Mock RunnableBinding with bound runnable."""

    def __init__(self, bound: Any, tools: list[MockTool] | None = None):
        self.bound = bound
        self.tools = tools or []

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        return "Bound result"

    def batch(self, *args: Any, **kwargs: Any) -> list[Any]:
        return ["Batch result"]


class MockCompiledGraph:
    """Mock LangGraph compiled graph."""

    def __init__(
        self,
        nodes: dict[str, Any],
        edges: list[tuple[str, str]] | None = None,
        tools: list[MockTool] | None = None,
    ):
        self.nodes = nodes
        self.edges = edges or []
        self.tools = tools or []


# =============================================================================
# Agent Definitions Based on GitHub Projects
# =============================================================================


@dataclass
class GitHubAgentConfig:
    """Configuration for a GitHub-sourced agent."""

    name: str
    repo: str
    description: str
    agent_factory: Any  # Callable that creates the agent


GITHUB_AGENTS: list[GitHubAgentConfig] = [
    # 1. langchain-ai/deepagents - Deep planning agent
    GitHubAgentConfig(
        name="DeepAgent",
        repo="langchain-ai/deepagents",
        description="Planning agent with filesystem backend and subagents",
        agent_factory=lambda: MockAgentExecutor(
            agent=MockAgent(
                MockChatAnthropic(
                    model_name="claude-sonnet-4-5-20250929", temperature=0.0
                )
            ),
            tools=[
                MockTool("plan", "Create a detailed plan for the task"),
                MockTool("filesystem_read", "Read files from the filesystem"),
                MockTool("filesystem_write", "Write files to the filesystem"),
                MockTool("spawn_subagent", "Spawn a subagent for subtasks"),
                MockTool("shell_execute", "Execute shell commands"),
            ],
            max_iterations=50,
        ),
    ),
    # 2. langchain-ai/agents-from-scratch - Email assistant
    GitHubAgentConfig(
        name="EmailAssistant",
        repo="langchain-ai/agents-from-scratch",
        description="Ambient email management agent with Gmail API",
        agent_factory=lambda: MockAgentExecutor(
            agent=MockAgent(MockChatOpenAI(model_name="gpt-4o", temperature=0.3)),
            tools=[
                MockTool("gmail_read", "Read emails from Gmail"),
                MockTool("gmail_send", "Send emails via Gmail"),
                MockTool("gmail_search", "Search emails in Gmail"),
                MockTool("calendar_check", "Check calendar availability"),
                MockTool("draft_response", "Draft email responses"),
            ],
            max_iterations=20,
        ),
    ),
    # 3. langchain-ai/streamlit-agent - Search chatbot
    GitHubAgentConfig(
        name="StreamlitSearchAgent",
        repo="langchain-ai/streamlit-agent",
        description="Search-enabled chatbot with memory",
        agent_factory=lambda: MockAgentExecutor(
            agent=MockAgent(
                MockChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)
            ),
            tools=[
                MockTool("web_search", "Search the web for information"),
                MockTool("calculator", "Perform mathematical calculations"),
            ],
            max_iterations=10,
        ),
    ),
    # 4. kabir12345/Agent-Experiments - Music recommendation agent
    GitHubAgentConfig(
        name="MusicRecommendationAgent",
        repo="kabir12345/Agent-Experiments",
        description="Music recommendation using Spotify API",
        agent_factory=lambda: MockAgentExecutor(
            agent=MockAgent(MockChatOpenAI(model_name="gpt-4", temperature=0.7)),
            tools=[
                MockTool("spotify_search", "Search Spotify for tracks"),
                MockTool("get_recommendations", "Get music recommendations"),
                MockTool("analyze_mood", "Analyze user mood for suggestions"),
            ],
            max_iterations=15,
        ),
    ),
    # 5. kabir12345/Agent-Experiments - Financial data agent
    GitHubAgentConfig(
        name="FinancialDataAgent",
        repo="kabir12345/Agent-Experiments",
        description="Financial data retrieval and analysis",
        agent_factory=lambda: MockAgentExecutor(
            agent=MockAgent(MockChatOpenAI(model_name="gpt-4", temperature=0.2)),
            tools=[
                MockTool("stock_price", "Get current stock prices"),
                MockTool("financial_news", "Get financial news"),
                MockTool("company_info", "Get company information"),
                MockTool("market_analysis", "Perform market analysis"),
            ],
            max_iterations=20,
        ),
    ),
    # 6. joaomdmoura/crewai - Role-playing autonomous agent
    GitHubAgentConfig(
        name="CrewAIAgent",
        repo="joaomdmoura/crewai",
        description="Role-playing autonomous AI agent for crew orchestration",
        agent_factory=lambda: MockAgentExecutor(
            agent=MockAgent(MockChatOpenAI(model_name="gpt-4-turbo", temperature=0.5)),
            tools=[
                MockTool("delegate_task", "Delegate task to another agent"),
                MockTool("ask_question", "Ask questions to gather information"),
                MockTool("web_search", "Search the web for research"),
                MockTool("file_writer", "Write content to files"),
            ],
            max_iterations=25,
        ),
    ),
    # 7. TransformerOptimus/SuperAGI - Autonomous AI framework agent
    GitHubAgentConfig(
        name="SuperAGIAgent",
        repo="TransformerOptimus/SuperAGI",
        description="Dev-first autonomous AI agent framework",
        agent_factory=lambda: MockAgentExecutor(
            agent=MockAgent(MockChatOpenAI(model_name="gpt-4", temperature=0.3)),
            tools=[
                MockTool("code_write", "Write code to files"),
                MockTool("code_read", "Read code from files"),
                MockTool("terminal_execute", "Execute terminal commands"),
                MockTool("google_search", "Search Google"),
                MockTool("github_toolkit", "Interact with GitHub"),
                MockTool("jira_toolkit", "Interact with Jira"),
            ],
            max_iterations=30,
        ),
    ),
    # 8. reworkd/AgentGPT - Web-based autonomous agent
    GitHubAgentConfig(
        name="AgentGPT",
        repo="reworkd/AgentGPT",
        description="Web-based AI agents with goal execution",
        agent_factory=lambda: MockAgentExecutor(
            agent=MockAgent(
                MockChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
            ),
            tools=[
                MockTool("create_task", "Create a new task"),
                MockTool("execute_task", "Execute a task"),
                MockTool("analyze_results", "Analyze task results"),
            ],
            max_iterations=15,
        ),
    ),
    # 9. assafelovic/gpt-researcher - Research agent
    GitHubAgentConfig(
        name="GPTResearcher",
        repo="assafelovic/gpt-researcher",
        description="Autonomous agent for comprehensive online research",
        agent_factory=lambda: MockAgentExecutor(
            agent=MockAgent(MockChatOpenAI(model_name="gpt-4o", temperature=0.4)),
            tools=[
                MockTool("web_scrape", "Scrape web pages for content"),
                MockTool("search_engine", "Search using multiple engines"),
                MockTool("summarize", "Summarize long documents"),
                MockTool("citation_finder", "Find citations and sources"),
                MockTool("report_writer", "Write research reports"),
            ],
            max_iterations=40,
        ),
    ),
    # 10. stepanogil/autonomous-hr-chatbot - HR agent
    GitHubAgentConfig(
        name="HRChatbot",
        repo="stepanogil/autonomous-hr-chatbot",
        description="Autonomous HR query agent with tools",
        agent_factory=lambda: MockAgentExecutor(
            agent=MockAgent(MockChatOpenAI(model_name="gpt-4", temperature=0.3)),
            tools=[
                MockTool("policy_lookup", "Look up HR policies"),
                MockTool("leave_calculator", "Calculate leave balance"),
                MockTool("benefits_info", "Get benefits information"),
                MockTool("org_chart", "Query organizational chart"),
            ],
            max_iterations=10,
        ),
    ),
    # 11. MineDojo/Voyager - Embodied Minecraft agent
    GitHubAgentConfig(
        name="Voyager",
        repo="MineDojo/Voyager",
        description="Open-ended embodied agent with LLMs",
        agent_factory=lambda: MockAgentExecutor(
            agent=MockAgent(MockChatOpenAI(model_name="gpt-4", temperature=0.0)),
            tools=[
                MockTool("execute_skill", "Execute a skill in the game"),
                MockTool("explore", "Explore the environment"),
                MockTool("craft_item", "Craft items in the game"),
                MockTool("skill_library", "Query skill library"),
            ],
            max_iterations=100,
        ),
    ),
    # 12. OpenBMB/XAgent - Complex task solving
    GitHubAgentConfig(
        name="XAgent",
        repo="OpenBMB/XAgent",
        description="Autonomous LLM agent for complex task solving",
        agent_factory=lambda: MockAgentExecutor(
            agent=MockAgent(MockChatOpenAI(model_name="gpt-4-turbo", temperature=0.2)),
            tools=[
                MockTool("plan_generation", "Generate task plans"),
                MockTool("code_executor", "Execute Python code"),
                MockTool("file_system", "File system operations"),
                MockTool("web_browser", "Browse web pages"),
                MockTool("shell_command", "Run shell commands"),
            ],
            max_iterations=50,
        ),
    ),
    # 13. microsoft/autogen - Multi-agent conversation
    GitHubAgentConfig(
        name="AutoGenAgent",
        repo="microsoft/autogen",
        description="Next-gen LLM application with multi-agent conversations",
        agent_factory=lambda: MockAgentExecutor(
            agent=MockAgent(MockChatOpenAI(model_name="gpt-4", temperature=0.5)),
            tools=[
                MockTool("code_writer", "Write code"),
                MockTool("code_executor", "Execute code"),
                MockTool("user_proxy", "Proxy for user interactions"),
            ],
            max_iterations=20,
        ),
    ),
    # 14. openbmb/agentverse - Multi-agent environment
    GitHubAgentConfig(
        name="AgentVerse",
        repo="openbmb/agentverse",
        description="Multi-agent environment for LLMs",
        agent_factory=lambda: MockAgentExecutor(
            agent=MockAgent(
                MockChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
            ),
            tools=[
                MockTool("communicate", "Communicate with other agents"),
                MockTool("vote", "Cast a vote in group decisions"),
                MockTool("debate", "Participate in debates"),
            ],
            max_iterations=30,
        ),
    ),
    # 15. chatchat-space/LangGraph-Chatchat - LangGraph chat
    GitHubAgentConfig(
        name="LangGraphChatchat",
        repo="chatchat-space/LangGraph-Chatchat",
        description="LangGraph-based chat implementation",
        agent_factory=lambda: MockCompiledGraph(
            nodes={
                "retrieve": MockLLM(model_name="gpt-4"),
                "generate": MockLLM(model_name="gpt-4"),
                "rerank": MockLLM(model_name="gpt-4"),
            },
            edges=[("retrieve", "rerank"), ("rerank", "generate")],
        ),
    ),
    # 16. Travel AI Agent with Cosmos DB
    GitHubAgentConfig(
        name="TravelAIAgent",
        repo="jonathanscholtes/Travel-AI-Agent-React-FastAPI-and-Cosmos-DB-Vector-Store",
        description="LangChain Agent using Azure Cosmos DB",
        agent_factory=lambda: MockAgentExecutor(
            agent=MockAgent(MockChatOpenAI(model_name="gpt-4o", temperature=0.5)),
            tools=[
                MockTool("search_destinations", "Search travel destinations"),
                MockTool("book_flight", "Book flights"),
                MockTool("hotel_search", "Search for hotels"),
                MockTool("itinerary_planner", "Plan travel itineraries"),
            ],
            max_iterations=15,
        ),
    ),
    # 17. LangGraph multi-agent system
    GitHubAgentConfig(
        name="LangGraphAgentSystem",
        repo="shamspias/langgraph-agent-system",
        description="Production-ready scalable multi-agent system",
        agent_factory=lambda: MockCompiledGraph(
            nodes={
                "supervisor": MockLLM(model_name="gpt-4o"),
                "researcher": MockLLM(model_name="gpt-4o-mini"),
                "coder": MockLLM(model_name="gpt-4o"),
                "reviewer": MockLLM(model_name="gpt-4o"),
            },
            edges=[
                ("supervisor", "researcher"),
                ("supervisor", "coder"),
                ("coder", "reviewer"),
            ],
        ),
    ),
    # 18. SQL Agent for data visualization
    GitHubAgentConfig(
        name="SQLVisualizationAgent",
        repo="EliasK93/LangChain-SQL-Agent-for-dynamic-data-visualization",
        description="LLM-based SQL Agent for dynamic queries",
        agent_factory=lambda: MockAgentExecutor(
            agent=MockAgent(MockChatOpenAI(model_name="gpt-4", temperature=0.0)),
            tools=[
                MockTool("sql_query", "Execute SQL queries"),
                MockTool("schema_lookup", "Get database schema"),
                MockTool("visualize_data", "Create data visualizations"),
            ],
            max_iterations=10,
        ),
    ),
    # 19. RAG Retail Inventory Agent
    GitHubAgentConfig(
        name="RetailInventoryAgent",
        repo="march038/RAG-Retail-Inventory-Agent",
        description="RAG agent for supermarket inventory database",
        agent_factory=lambda: MockAgentExecutor(
            agent=MockAgent(MockChatOpenAI(model_name="gpt-4", temperature=0.2)),
            tools=[
                MockTool("inventory_search", "Search inventory"),
                MockTool("stock_check", "Check stock levels"),
                MockTool("reorder_alert", "Generate reorder alerts"),
                MockTool("price_lookup", "Look up product prices"),
            ],
            max_iterations=10,
        ),
    ),
    # 20. Multi-modal LangChain chatbot
    GitHubAgentConfig(
        name="MultiModalChatbot",
        repo="sachs7/multi-modal-langchain-chatbot",
        description="Multi-modal chatbot with RAG and DALL-E",
        agent_factory=lambda: MockAgentExecutor(
            agent=MockAgent(
                MockChatOpenAI(model_name="gpt-4-vision-preview", temperature=0.7)
            ),
            tools=[
                MockTool("rag_search", "Search documents with RAG"),
                MockTool("dalle_generate", "Generate images with DALL-E"),
                MockTool("paper_search", "Search papers on Paperswithcode"),
                MockTool("image_analyze", "Analyze images"),
            ],
            max_iterations=15,
        ),
    ),
    # 21. FastAPI streaming agent
    GitHubAgentConfig(
        name="StreamingAgent",
        repo="keshe4ka/fastapi-langchain-agent-streaming-example",
        description="FastAPI app with streaming agents and Tavily",
        agent_factory=lambda: MockAgentExecutor(
            agent=MockAgent(MockChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)),
            tools=[
                MockTool("tavily_search", "Search with Tavily API"),
            ],
            max_iterations=10,
        ),
    ),
    # 22. Healthcare chatbot
    GitHubAgentConfig(
        name="HealthGenieAgent",
        repo="rahulkr43/A-Multimodel-Healthcare-Chatbot-Healthgenie",
        description="Agentic RAG for healthcare assistance",
        agent_factory=lambda: MockAgentExecutor(
            agent=MockAgent(MockChatOpenAI(model_name="gpt-4", temperature=0.3)),
            tools=[
                MockTool("symptom_checker", "Check symptoms"),
                MockTool("drug_lookup", "Look up drug information"),
                MockTool("appointment_scheduler", "Schedule appointments"),
                MockTool("medical_history", "Query medical history"),
            ],
            max_iterations=15,
        ),
    ),
    # 23. LangChain autonomous agents system
    GitHubAgentConfig(
        name="AutonomousAgents",
        repo="riolaf05/langchain-autonomous-agents",
        description="System of autonomous agents powered by LLMs",
        agent_factory=lambda: MockAgentExecutor(
            agent=MockAgent(
                MockChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
            ),
            tools=[
                MockTool("task_queue", "Manage task queue"),
                MockTool("memory_store", "Store and retrieve memories"),
                MockTool("goal_tracker", "Track agent goals"),
            ],
            max_iterations=25,
        ),
    ),
    # 24. Claude-based agent (local model)
    GitHubAgentConfig(
        name="ClaudeLocalAgent",
        repo="anthropic/claude-agent-example",
        description="Claude-based agent with Anthropic API",
        agent_factory=lambda: MockAgentExecutor(
            agent=MockAgent(
                MockChatAnthropic(
                    model_name="claude-3-5-sonnet-latest", temperature=0.5
                )
            ),
            tools=[
                MockTool("analyze_code", "Analyze code snippets"),
                MockTool("explain_concept", "Explain complex concepts"),
                MockTool("generate_code", "Generate code solutions"),
            ],
            max_iterations=20,
        ),
    ),
    # 25. Ollama local agent
    GitHubAgentConfig(
        name="OllamaLocalAgent",
        repo="local/ollama-agent",
        description="Local LLM agent with Ollama",
        agent_factory=lambda: MockAgentExecutor(
            agent=MockAgent(MockOllama(model_name="llama3.2", temperature=0.7)),
            tools=[
                MockTool("local_search", "Search local documents"),
                MockTool("text_summary", "Summarize text"),
            ],
            max_iterations=15,
        ),
    ),
]


# =============================================================================
# Test Cases
# =============================================================================


class TestGitHubAgents:
    """Test AgentFacts introspection on GitHub-sourced agents."""

    @pytest.mark.parametrize(
        "config", GITHUB_AGENTS, ids=[c.name for c in GITHUB_AGENTS]
    )
    def test_introspect_github_agent(self, config: GitHubAgentConfig) -> None:
        """Test introspection of each GitHub agent."""
        agent = config.agent_factory()

        # Use introspect_any to auto-detect agent type
        baseline_model, capabilities, constraints = introspect_any(agent)

        # Verify we extracted something meaningful
        if isinstance(agent, MockCompiledGraph):
            # LangGraph agents might not have a baseline model in nodes
            # but should have node capabilities
            assert (
                len(capabilities) > 0
            ), f"{config.name}: Should have node capabilities"
        else:
            # AgentExecutor should have baseline model and capabilities
            assert (
                baseline_model is not None
            ), f"{config.name}: Should have baseline model"
            assert (
                len(capabilities) > 0
            ), f"{config.name}: Should have capabilities from tools"

        print(f"\n{'='*60}")
        print(f"Agent: {config.name}")
        print(f"Repo: {config.repo}")
        print(f"Description: {config.description}")
        print("-" * 60)
        if baseline_model:
            print(f"Model: {baseline_model.name}")
            print(f"Provider: {baseline_model.provider.value}")
            print(f"Temperature: {baseline_model.temperature}")
        print(f"Capabilities ({len(capabilities)}):")
        for cap in capabilities[:5]:  # Show first 5
            print(f"  - {cap.name}: {cap.risk_level}")
        if len(capabilities) > 5:
            print(f"  ... and {len(capabilities) - 5} more")
        print(f"Constraints: max_iterations={constraints.max_iterations}")

    @pytest.mark.parametrize(
        "config", GITHUB_AGENTS, ids=[c.name for c in GITHUB_AGENTS]
    )
    def test_agentfacts_from_introspection(self, config: GitHubAgentConfig) -> None:
        """Test creating AgentFacts from introspected data."""
        agent = config.agent_factory()
        key_pair = KeyPair.generate()

        # Introspect the agent
        baseline_model, capabilities, constraints = introspect_any(agent)

        # Create AgentFacts manually with introspected data
        facts = AgentFacts(
            name=config.name,
            description=config.description,
            key_pair=key_pair,
        )

        # Add model and capabilities
        if baseline_model:
            facts.metadata.agent.model = baseline_model
        facts.metadata.agent.capabilities = capabilities
        facts.metadata.policy.constraints = constraints

        # Sign and verify
        facts.sign()
        result = facts.verify()

        assert result.valid, f"{config.name}: Signature verification failed"
        assert facts.did is not None, f"{config.name}: Should have DID"

        # Serialize and deserialize
        json_str = facts.to_json()
        assert len(json_str) > 100, f"{config.name}: Should serialize to JSON"

    def test_provider_detection(self) -> None:
        """Test that provider detection works correctly."""
        test_cases = [
            (MockChatOpenAI(model_name="gpt-4"), ModelProvider.OPENAI),
            (MockChatOpenAI(model_name="gpt-3.5-turbo"), ModelProvider.OPENAI),
            (MockChatAnthropic(model_name="claude-3-opus"), ModelProvider.ANTHROPIC),
            (MockChatGoogleGenerativeAI(model_name="gemini-pro"), ModelProvider.GOOGLE),
            # llama model names are detected as META provider even when hosted locally
            # For true local detection, endpoint-based detection is more reliable
            (MockOllama(model_name="llama3.2"), ModelProvider.META),
        ]

        for llm, expected_provider in test_cases:
            model = introspect_llm(llm)
            assert (
                model.provider == expected_provider
            ), f"Expected {expected_provider} for {llm.model_name}, got {model.provider}"

    def test_tool_risk_assessment(self) -> None:
        """Test that tool risk levels are assessed correctly."""
        tools = [
            MockTool("shell_execute", "Execute shell commands"),  # high
            MockTool("code_executor", "Execute Python code"),  # high
            MockTool("http_request", "Make HTTP requests"),  # medium
            MockTool("file_read", "Read files"),  # medium
            MockTool("add_numbers", "Add two numbers together"),  # low
        ]

        capabilities = introspect_tools(tools)

        assert capabilities[0].risk_level == "high"  # shell_execute
        assert capabilities[1].risk_level == "high"  # code_executor
        assert capabilities[2].risk_level == "medium"  # http_request
        assert capabilities[3].risk_level == "medium"  # file_read
        assert capabilities[4].risk_level == "low"  # add_numbers

    def test_langgraph_introspection(self) -> None:
        """Test LangGraph compiled graph introspection."""
        graph = MockCompiledGraph(
            nodes={
                "__start__": None,
                "agent": MockLLM(model_name="gpt-4o"),
                "tools": MockLLM(model_name="gpt-4o-mini"),
                "__end__": None,
            },
            edges=[("__start__", "agent"), ("agent", "tools"), ("tools", "__end__")],
            tools=[MockTool("calculator", "Math calculations")],
        )

        baseline_model, capabilities, constraints = introspect_langgraph(graph)

        # Should find the LLM in nodes
        assert baseline_model is not None
        assert baseline_model.name == "gpt-4o"

        # Should have node capabilities (excluding special nodes) + tool
        node_caps = [c for c in capabilities if c.name.startswith("node:")]
        assert len(node_caps) == 2  # agent and tools nodes

        # Should also include the tool
        tool_caps = [c for c in capabilities if c.name == "calculator"]
        assert len(tool_caps) == 1


class TestAgentFactsIntegration:
    """Integration tests for full AgentFacts workflow."""

    def test_full_workflow_for_all_agents(self) -> None:
        """Test complete AgentFacts workflow for all agents."""
        results = []

        for config in GITHUB_AGENTS:
            agent = config.agent_factory()
            key_pair = KeyPair.generate()

            # Introspect
            baseline_model, capabilities, constraints = introspect_any(agent)

            # Create AgentFacts
            facts = AgentFacts(
                name=config.name,
                description=config.description,
                key_pair=key_pair,
            )
            if baseline_model:
                facts.metadata.agent.model = baseline_model
            facts.metadata.agent.capabilities = capabilities
            facts.metadata.policy.constraints = constraints

            # Sign
            facts.sign()

            # Verify
            result = facts.verify()

            results.append(
                {
                    "name": config.name,
                    "repo": config.repo,
                    "valid": result.valid,
                    "model": baseline_model.name if baseline_model else "N/A",
                    "provider": (
                        baseline_model.provider.value if baseline_model else "N/A"
                    ),
                    "capabilities": len(capabilities),
                }
            )

        # Print summary
        print("\n" + "=" * 80)
        print("AGENTFACTS GITHUB AGENT COMPATIBILITY REPORT")
        print("=" * 80)
        print(f"{'Agent':<30} {'Model':<25} {'Provider':<12} {'Caps':<6} {'Valid'}")
        print("-" * 80)

        for r in results:
            status = "✓" if r["valid"] else "✗"
            line = (
                f"{r['name']:<30} {r['model']:<25} {r['provider']:<12} "
                f"{r['capabilities']:<6} {status}"
            )
            print(line)

        print("-" * 80)
        valid_count = sum(1 for r in results if r["valid"])
        print(f"Total: {len(results)} agents, {valid_count} verified successfully")
        print("=" * 80)

        # All should pass
        assert all(r["valid"] for r in results), "All agents should verify successfully"
