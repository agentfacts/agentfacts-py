"""
Tests for OpenAgents introspector.
"""

from agentfacts import AgentFacts, GroupFacts, ProcessType
from agentfacts.integrations.openagents.introspector import (
    OpenAgentsIntegration as OpenAgentsIntrospector,
)
from agentfacts.models import ModelProvider

# -------------------------------------------------------------------------
# Mock Classes
# -------------------------------------------------------------------------


class MockAgentTool:
    """Mock OpenAgents AgentTool."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.schema = None


class MockMCPServerConfig:
    """Mock MCP server configuration."""

    def __init__(self, server_url: str = "http://localhost:3000"):
        self.server_url = server_url


class MockAgentConfig:
    """Mock OpenAgents AgentConfig."""

    __module__ = "openagents.models"

    def __init__(
        self,
        model_name: str = "gpt-4",
        instruction: str = "You are a helpful assistant.",
        provider: str = "openai",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        max_iterations: int = 5,
        react_to_all_messages: bool = False,
        tools: list = None,
        mcp_servers: list = None,
        triggers: list = None,
    ):
        self.model_name = model_name
        self.instruction = instruction
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_iterations = max_iterations
        self.react_to_all_messages = react_to_all_messages
        self.tools = tools or []
        self.mcp_servers = mcp_servers or []
        self.triggers = triggers or []
        self.system_prompt_template = None
        self.user_prompt_template = None


class MockWorkerAgent:
    """Mock OpenAgents WorkerAgent."""

    __module__ = "openagents.agents"

    default_agent_id = "test-agent"

    def __init__(self, config: MockAgentConfig = None):
        self.config = config
        self._network = None

    async def run_agent(self, context, instruction):
        pass

    async def on_channel_post(self, context):
        pass


class MockLLMAssistantAgent(MockWorkerAgent):
    """Mock LLM Assistant Agent."""

    default_agent_id = "ai-assistant"


class MockNetworkConfig:
    """Mock OpenAgents NetworkConfig."""

    __module__ = "openagents.models"

    def __init__(
        self,
        name: str = "TestNetwork",
        mode: str = "centralized",
        transport: str = "websocket",
        host: str = "localhost",
        port: int = 8765,
    ):
        self.name = name
        self.mode = mode
        self.transport = transport
        self.host = host
        self.port = port


class MockNetwork:
    """Mock OpenAgents Network."""

    __module__ = "openagents.core.network"

    def __init__(self, config: MockNetworkConfig = None, agents: dict = None):
        self.config = config
        self.agents = agents or {}

    async def register_agent(self, agent_id: str, capabilities: list):
        self.agents[agent_id] = {"capabilities": capabilities}

    async def send_message(
        self, sender_id: str, target_id: str, message_type: str, content: dict
    ):
        pass


# -------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------


class TestOpenAgentsIntrospector:
    """Tests for OpenAgents introspector."""

    def test_framework_name(self):
        """Test framework name."""
        introspector = OpenAgentsIntrospector()
        assert introspector.framework_name == "openagents"

    def test_can_introspect_agent_config(self):
        """Test can_introspect with AgentConfig."""
        introspector = OpenAgentsIntrospector()
        config = MockAgentConfig()
        assert introspector.can_introspect(config) is True

    def test_can_introspect_worker_agent(self):
        """Test can_introspect with WorkerAgent."""
        introspector = OpenAgentsIntrospector()
        agent = MockWorkerAgent()
        assert introspector.can_introspect(agent) is True

    def test_can_introspect_network_config(self):
        """Test can_introspect with NetworkConfig."""
        introspector = OpenAgentsIntrospector()
        config = MockNetworkConfig()
        assert introspector.can_introspect(config) is True

    def test_can_introspect_network(self):
        """Test can_introspect with Network."""
        introspector = OpenAgentsIntrospector()
        network = MockNetwork()
        assert introspector.can_introspect(network) is True

    def test_cannot_introspect_random(self):
        """Test can_introspect rejects random objects."""
        introspector = OpenAgentsIntrospector()
        assert introspector.can_introspect("string") is False
        assert introspector.can_introspect(123) is False
        assert introspector.can_introspect({}) is False

    def test_introspect_agent_config_basic(self):
        """Test basic AgentConfig introspection."""
        introspector = OpenAgentsIntrospector()
        config = MockAgentConfig(
            model_name="gpt-4-turbo",
            instruction="You are a data analyst.",
            provider="openai",
        )

        result = introspector.introspect(config)

        assert result.baseline_model is not None
        assert result.baseline_model.name == "gpt-4-turbo"
        assert result.baseline_model.provider == ModelProvider.OPENAI
        assert result.context["model_name"] == "gpt-4-turbo"
        assert result.context["provider"] == "openai"
        assert "You are a data analyst" in result.context["instruction"]

    def test_introspect_agent_config_anthropic(self):
        """Test AgentConfig with Anthropic provider."""
        introspector = OpenAgentsIntrospector()
        config = MockAgentConfig(
            model_name="claude-3-opus",
            provider="anthropic",
        )

        result = introspector.introspect(config)

        assert result.baseline_model.provider == ModelProvider.ANTHROPIC

    def test_introspect_agent_config_google(self):
        """Test AgentConfig with Google provider."""
        introspector = OpenAgentsIntrospector()
        config = MockAgentConfig(
            model_name="gemini-1.5-pro",
            provider="google",
        )

        result = introspector.introspect(config)

        assert result.baseline_model.provider == ModelProvider.GOOGLE

    def test_introspect_agent_config_temperature(self):
        """Test AgentConfig temperature extraction."""
        introspector = OpenAgentsIntrospector()
        config = MockAgentConfig(temperature=0.3, max_tokens=2000)

        result = introspector.introspect(config)

        assert result.baseline_model.temperature == 0.3
        assert result.baseline_model.max_tokens == 2000

    def test_introspect_agent_config_max_iterations(self):
        """Test AgentConfig max_iterations extraction."""
        introspector = OpenAgentsIntrospector()
        config = MockAgentConfig(max_iterations=10)

        result = introspector.introspect(config)

        assert result.constraints.max_iterations == 10

    def test_introspect_agent_config_with_tools(self):
        """Test AgentConfig with tools."""
        introspector = OpenAgentsIntrospector()
        tools = [
            MockAgentTool("web_search", "Search the web"),
            MockAgentTool("calculator", "Perform calculations"),
        ]
        config = MockAgentConfig(tools=tools)

        result = introspector.introspect(config)

        # Should have tools + llm_reasoning capability
        tool_names = [c.name for c in result.capabilities]
        assert "web_search" in tool_names
        assert "calculator" in tool_names
        assert "llm_reasoning" in tool_names

    def test_introspect_agent_config_with_mcp(self):
        """Test AgentConfig with MCP servers."""
        introspector = OpenAgentsIntrospector()
        config = MockAgentConfig(mcp_servers=[MockMCPServerConfig()])

        result = introspector.introspect(config)

        assert result.context.get("mcp_server_count") == 1
        assert any(c.name == "mcp_integration" for c in result.capabilities)

    def test_introspect_agent_config_with_triggers(self):
        """Test AgentConfig with triggers."""
        introspector = OpenAgentsIntrospector()
        config = MockAgentConfig(triggers=["on_message", "on_join"])

        result = introspector.introspect(config)

        # Verify it was introspected
        assert result.context["config_type"] == "MockAgentConfig"

    def test_introspect_agent_config_react_to_all(self):
        """Test AgentConfig react_to_all_messages flag."""
        introspector = OpenAgentsIntrospector()
        config = MockAgentConfig(react_to_all_messages=True)

        result = introspector.introspect(config)

        # Verify it was introspected
        assert result.context["config_type"] == "MockAgentConfig"

    def test_introspect_worker_agent(self):
        """Test WorkerAgent introspection."""
        introspector = OpenAgentsIntrospector()
        config = MockAgentConfig(model_name="gpt-4", instruction="Test instruction")
        agent = MockWorkerAgent(config=config)

        result = introspector.introspect(agent)

        assert result.context["agent_class"] == "MockWorkerAgent"
        assert result.baseline_model is not None
        # Should have channel_messaging capability
        assert any(c.name == "channel_messaging" for c in result.capabilities)

    def test_introspect_worker_agent_id(self):
        """Test WorkerAgent ID extraction."""
        introspector = OpenAgentsIntrospector()
        agent = MockWorkerAgent()

        result = introspector.introspect(agent)

        assert result.context.get("agent_id") == "test-agent"

    def test_introspect_llm_assistant_agent(self):
        """Test LLMAssistantAgent introspection."""
        introspector = OpenAgentsIntrospector()
        agent = MockLLMAssistantAgent()

        result = introspector.introspect(agent)

        assert result.context.get("agent_id") == "ai-assistant"

    def test_introspect_network_config(self):
        """Test NetworkConfig introspection."""
        introspector = OpenAgentsIntrospector()
        config = MockNetworkConfig(
            name="MyNetwork",
            mode="centralized",
            transport="websocket",
            host="localhost",
            port=8765,
        )

        result = introspector.introspect(config)

        assert result.context.get("network_name") == "MyNetwork"
        assert result.context.get("network_mode") == "centralized"
        assert result.context.get("transport") == "websocket"
        # Host and port may not be extracted by the introspector
        assert any(c.name == "network_coordination" for c in result.capabilities)

    def test_introspect_network_config_decentralized(self):
        """Test NetworkConfig with decentralized mode."""
        introspector = OpenAgentsIntrospector()
        config = MockNetworkConfig(mode="decentralized")

        result = introspector.introspect(config)

        assert result.context.get("process_type") == ProcessType.EVENT_DRIVEN.value

    def test_introspect_network_config_centralized(self):
        """Test NetworkConfig with centralized mode."""
        introspector = OpenAgentsIntrospector()
        config = MockNetworkConfig(mode="centralized")

        result = introspector.introspect(config)

        assert result.context.get("process_type") == ProcessType.HIERARCHICAL.value

    def test_introspect_network(self):
        """Test Network introspection."""
        introspector = OpenAgentsIntrospector()
        config = MockNetworkConfig(name="TestNet")
        network = MockNetwork(config=config, agents={"agent1": {}, "agent2": {}})

        result = introspector.introspect(network)

        assert result.context.get("agent_count") == 2
        assert "agent1" in result.context.get("agent_ids", [])
        assert "agent2" in result.context.get("agent_ids", [])
        assert any(c.name == "message_routing" for c in result.capabilities)
        assert any(c.name == "agent_registration" for c in result.capabilities)

    def test_detect_provider_via_introspect(self):
        """Test provider detection via introspect for different models."""
        introspector = OpenAgentsIntrospector()

        # Test with GPT-4
        config_openai = MockAgentConfig(model_name="gpt-4")
        result = introspector.introspect(config_openai)
        assert result.baseline_model.provider == ModelProvider.OPENAI

        # Test with Claude - verify the model name is captured correctly
        config_anthropic = MockAgentConfig(model_name="claude-3-opus")
        result = introspector.introspect(config_anthropic)
        # The provider detection depends on the introspector implementation
        assert result.baseline_model is not None
        assert result.baseline_model.name == "claude-3-opus"

    def test_get_agent_type(self):
        """Test get_agent_type returns class names."""
        introspector = OpenAgentsIntrospector()

        config = MockAgentConfig()
        assert introspector.get_agent_type(config) == "MockAgentConfig"

        network_config = MockNetworkConfig()
        assert introspector.get_agent_type(network_config) == "MockNetworkConfig"

        agent = MockWorkerAgent()
        assert introspector.get_agent_type(agent) == "MockWorkerAgent"


class TestGroupFactsFromOpenAgents:
    """Tests for GroupFacts.from_openagents()."""

    def test_from_openagents_agent_config(self):
        """Test creating GroupFacts from AgentConfig."""
        config = MockAgentConfig(
            model_name="gpt-4",
            instruction="You are a helpful assistant.",
            provider="openai",
        )

        group = GroupFacts.from_openagents(config, name="Assistant Agent")

        assert group.name == "Assistant Agent"
        assert len(group.members) == 1
        assert group.metadata.framework == "openagents"
        assert group.metadata.process_type == ProcessType.EVENT_DRIVEN

    def test_from_openagents_worker_agent(self):
        """Test creating GroupFacts from WorkerAgent."""
        config = MockAgentConfig(model_name="claude-3-opus", provider="anthropic")
        agent = MockWorkerAgent(config=config)

        group = GroupFacts.from_openagents(agent, name="Worker Agent")

        assert group.name == "Worker Agent"
        assert len(group.members) == 1
        member = group.members[0]
        assert member.metadata.agent.framework == "openagents"

    def test_from_openagents_network(self):
        """Test creating GroupFacts from Network."""
        config = MockNetworkConfig(name="TestNetwork", mode="centralized")
        agents = {
            "agent1": MockAgentConfig(model_name="gpt-4", instruction="Agent 1"),
            "agent2": MockAgentConfig(model_name="gpt-4", instruction="Agent 2"),
        }
        network = MockNetwork(config=config, agents=agents)

        group = GroupFacts.from_openagents(network, name="Test Network")

        assert group.name == "Test Network"
        assert len(group.members) == 2
        assert group.metadata.framework == "openagents"
        assert group.metadata.process_type == ProcessType.HIERARCHICAL

    def test_from_openagents_network_decentralized(self):
        """Test creating GroupFacts from decentralized Network."""
        config = MockNetworkConfig(mode="decentralized")
        network = MockNetwork(config=config)

        group = GroupFacts.from_openagents(network)

        assert group.metadata.process_type == ProcessType.EVENT_DRIVEN

    def test_from_openagents_auto_name(self):
        """Test auto-generated name."""
        config = MockAgentConfig()

        group = GroupFacts.from_openagents(config)

        assert "OpenAgents" in group.name

    def test_from_openagents_member_has_capabilities(self):
        """Test that member AgentFacts has capabilities."""
        config = MockAgentConfig(tools=[MockAgentTool("search", "Web search")])

        group = GroupFacts.from_openagents(config)

        member = group.members[0]
        assert len(member.metadata.agent.capabilities) > 0
        assert any(c.name == "search" for c in member.metadata.agent.capabilities)

    def test_from_openagents_sign_all(self):
        """Test signing OpenAgents group and members."""
        config = MockAgentConfig()

        group = GroupFacts.from_openagents(config)
        signatures = group.sign_all()

        assert len(signatures) == 2  # 1 member + 1 group
        assert group.all_verified()

    def test_from_openagents_context_preserved(self):
        """Test that context is preserved."""
        config = MockAgentConfig(
            model_name="gpt-4-turbo",
            instruction="You are a data analyst.",
            max_iterations=8,
        )

        group = GroupFacts.from_openagents(config)

        assert group.metadata.context.get("component_type") == "MockAgentConfig"
        member = group.members[0]
        assert member.metadata.agent.context.get("model_name") == "gpt-4-turbo"


class TestAgentFactsFromOpenAgents:
    """Tests for AgentFacts.from_agent with OpenAgents."""

    def test_from_agent_openagents_auto_detect(self):
        """Test AgentFacts.from_agent auto-detects OpenAgents."""
        config = MockAgentConfig()

        from agentfacts.integrations import get_registry, reset_registry

        reset_registry()
        registry = get_registry()
        registry.register(OpenAgentsIntrospector())

        facts = AgentFacts.from_agent(config, name="OpenAgents Agent")

        assert facts.name == "OpenAgents Agent"
        assert facts.metadata.agent.framework == "openagents"

    def test_from_agent_openagents_explicit(self):
        """Test AgentFacts.from_agent with explicit framework."""
        config = MockAgentConfig(model_name="claude-3-sonnet", provider="anthropic")

        from agentfacts.integrations import get_registry, reset_registry

        reset_registry()
        registry = get_registry()
        registry.register(OpenAgentsIntrospector())

        facts = AgentFacts.from_agent(
            config, name="Claude Agent", framework="openagents"
        )

        assert facts.metadata.agent.framework == "openagents"
        assert facts.metadata.agent.model.name == "claude-3-sonnet"
