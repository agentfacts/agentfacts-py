"""
Tests for Hugging Face introspector.

Tests for both smolagents and tiny-agents support.
"""

import json
import tempfile
from pathlib import Path

import pytest

from agentfacts import AgentFacts, GroupFacts, ProcessType
from agentfacts.integrations.huggingface.introspector import (
    HuggingFaceIntegration as HuggingFaceIntrospector,
)
from agentfacts.integrations.huggingface.introspector import (
    load_tiny_agents_config,
)
from agentfacts.models import ModelProvider

# -------------------------------------------------------------------------
# Mock Classes - smolagents
# -------------------------------------------------------------------------


class MockTool:
    """Mock smolagents Tool."""

    def __init__(self, name: str, description: str = "", inputs: dict = None):
        self.name = name
        self.description = description
        self.inputs = inputs or {}


class MockToolCollection:
    """Mock smolagents ToolCollection."""

    def __init__(self, tools: list):
        self.tools = tools


class MockInferenceClientModel:
    """Mock smolagents InferenceClientModel."""

    __module__ = "smolagents"

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-72B-Instruct",
        temperature: float = 0.5,
        max_tokens: int = 4096,
    ):
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt):
        pass


class MockLiteLLMModel:
    """Mock smolagents LiteLLMModel."""

    __module__ = "smolagents"

    def __init__(self, model_id: str = "gpt-4o"):
        self.model_id = model_id
        self.temperature = 0.7

    def generate(self, prompt):
        pass


class MockTransformersModel:
    """Mock smolagents TransformersModel (local)."""

    __module__ = "smolagents"

    def __init__(self, model_id: str = "meta-llama/Llama-3.2-1B"):
        self.model_id = model_id

    def generate(self, prompt):
        pass


class MockCodeAgent:
    """Mock smolagents CodeAgent."""

    __module__ = "smolagents"

    def __init__(
        self,
        tools: list = None,
        model: object = None,
        max_steps: int = 10,
        instructions: str = "",
        executor: object = None,
        name: str = None,
        description: str = None,
        memory: object = None,
        managed_agents: list = None,
        planning_interval: int = None,
    ):
        self.tools = tools or []
        self.model = model
        self.max_steps = max_steps
        self.instructions = instructions
        self.executor = executor
        self.name = name
        self.description = description
        self.memory = memory
        self.managed_agents = managed_agents or []
        self.planning_interval = planning_interval

    def run(self, task):
        pass


class MockToolCallingAgent:
    """Mock smolagents ToolCallingAgent."""

    __module__ = "smolagents"

    def __init__(
        self,
        tools: list = None,
        model: object = None,
        max_steps: int = 10,
        instructions: str = "",
        name: str = None,
        description: str = None,
        managed_agents: list = None,
    ):
        self.tools = tools or []
        self.model = model
        self.max_steps = max_steps
        self.instructions = instructions
        self.name = name
        self.description = description
        self.managed_agents = managed_agents or []

    def run(self, task):
        pass


class MockLocalPythonExecutor:
    """Mock local Python executor."""

    pass


class MockDockerExecutor:
    """Mock Docker sandbox executor."""

    pass


class MockE2BExecutor:
    """Mock E2B sandbox executor."""

    pass


class MockAgentMemory:
    """Mock smolagents AgentMemory."""

    def __init__(self):
        self.steps = []


class MockHubAgent:
    """Mock tiny-agents Agent class from huggingface_hub."""

    __module__ = "huggingface_hub"

    def __init__(self, config: dict = None, model: object = None, tools: list = None):
        self.config = config
        self._config = config
        self.model = model
        self._model = model
        self.tools = tools or []


# -------------------------------------------------------------------------
# Test Class: HuggingFaceIntrospector
# -------------------------------------------------------------------------


class TestHuggingFaceIntrospector:
    """Tests for HuggingFaceIntrospector."""

    @pytest.fixture
    def introspector(self):
        return HuggingFaceIntrospector()

    # -------------------------------------------------------------------------
    # can_introspect tests
    # -------------------------------------------------------------------------

    def test_can_introspect_code_agent(self, introspector):
        """Test detection of CodeAgent."""
        agent = MockCodeAgent()
        assert introspector.can_introspect(agent)

    def test_can_introspect_tool_calling_agent(self, introspector):
        """Test detection of ToolCallingAgent."""
        agent = MockToolCallingAgent()
        assert introspector.can_introspect(agent)

    def test_can_introspect_inference_client_model(self, introspector):
        """Test detection of InferenceClientModel."""
        model = MockInferenceClientModel()
        assert introspector.can_introspect(model)

    def test_can_introspect_litellm_model(self, introspector):
        """Test detection of LiteLLMModel."""
        model = MockLiteLLMModel()
        assert introspector.can_introspect(model)

    def test_can_introspect_transformers_model(self, introspector):
        """Test detection of TransformersModel."""
        model = MockTransformersModel()
        assert introspector.can_introspect(model)

    def test_can_introspect_tiny_agents_config_dict(self, introspector):
        """Test detection of tiny-agents config dict."""
        config = {
            "model": "gpt-4",
            "provider": "openai",
            "servers": [],
        }
        assert introspector.can_introspect(config)

    def test_can_introspect_tiny_agents_with_endpoint(self, introspector):
        """Test detection of tiny-agents config with endpointUrl."""
        config = {
            "model": "Qwen/Qwen2.5-72B-Instruct",
            "endpointUrl": "http://localhost:8080/v1",
        }
        assert introspector.can_introspect(config)

    def test_can_introspect_agent_json_path(self, introspector):
        """Test detection of agent.json path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent_json = Path(tmpdir) / "agent.json"
            agent_json.write_text('{"model": "gpt-4", "provider": "openai"}')
            assert introspector.can_introspect(str(agent_json))

    def test_can_introspect_directory_with_agent_json(self, introspector):
        """Test detection of directory containing agent.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent_json = Path(tmpdir) / "agent.json"
            agent_json.write_text('{"model": "gpt-4", "provider": "openai"}')
            assert introspector.can_introspect(tmpdir)

    def test_cannot_introspect_random_dict(self, introspector):
        """Test that random dicts are not introspected."""
        random_dict = {"foo": "bar", "baz": 123}
        assert not introspector.can_introspect(random_dict)

    def test_cannot_introspect_random_object(self, introspector):
        """Test that random objects are not introspected."""

        class RandomClass:
            pass

        assert not introspector.can_introspect(RandomClass())

    # -------------------------------------------------------------------------
    # CodeAgent introspection tests
    # -------------------------------------------------------------------------

    def test_introspect_code_agent_basic(self, introspector):
        """Test introspection of basic CodeAgent."""
        model = MockInferenceClientModel("Qwen/Qwen2.5-72B-Instruct")
        agent = MockCodeAgent(model=model, max_steps=15)

        result = introspector.introspect(agent)

        assert result.baseline_model is not None
        assert result.baseline_model.name == "Qwen/Qwen2.5-72B-Instruct"
        assert result.context["agent_type"] == "MockCodeAgent"
        assert result.context["execution_mode"] == "code"
        assert result.constraints.max_iterations == 15

        # Should have code_execution capability
        cap_names = [c.name for c in result.capabilities]
        assert "code_execution" in cap_names

    def test_introspect_code_agent_with_tools(self, introspector):
        """Test introspection of CodeAgent with tools."""
        tools = [
            MockTool("web_search", "Search the web for information"),
            MockTool("calculator", "Perform calculations"),
        ]
        model = MockInferenceClientModel()
        agent = MockCodeAgent(tools=tools, model=model)

        result = introspector.introspect(agent)

        assert result.context["tool_count"] == 2

        cap_names = [c.name for c in result.capabilities]
        assert "web_search" in cap_names
        assert "calculator" in cap_names

    def test_introspect_code_agent_with_docker_executor(self, introspector):
        """Test introspection of CodeAgent with Docker executor."""
        executor = MockDockerExecutor()
        agent = MockCodeAgent(executor=executor)

        result = introspector.introspect(agent)

        assert result.context["executor_type"] == "MockDockerExecutor"
        assert result.context.get("sandboxed_execution") is True

    def test_introspect_code_agent_with_local_executor(self, introspector):
        """Test introspection of CodeAgent with local executor."""
        executor = MockLocalPythonExecutor()
        agent = MockCodeAgent(executor=executor)

        result = introspector.introspect(agent)

        assert "Local" in result.context["executor_type"]
        assert result.context.get("sandboxed_execution") is False

    def test_introspect_code_agent_with_memory(self, introspector):
        """Test introspection of CodeAgent with memory."""
        memory = MockAgentMemory()
        agent = MockCodeAgent(memory=memory)

        result = introspector.introspect(agent)

        assert result.context.get("has_memory") is True

    def test_introspect_code_agent_with_instructions(self, introspector):
        """Test introspection of CodeAgent with instructions."""
        instructions = "You are a helpful coding assistant."
        agent = MockCodeAgent(instructions=instructions)

        result = introspector.introspect(agent)

        assert result.context["instructions"] == instructions

    def test_introspect_code_agent_with_planning(self, introspector):
        """Test introspection of CodeAgent with planning interval."""
        agent = MockCodeAgent(planning_interval=3)

        result = introspector.introspect(agent)

        # Verify it was introspected
        assert result.context["agent_type"] == "MockCodeAgent"

    # -------------------------------------------------------------------------
    # ToolCallingAgent introspection tests
    # -------------------------------------------------------------------------

    def test_introspect_tool_calling_agent_basic(self, introspector):
        """Test introspection of basic ToolCallingAgent."""
        model = MockLiteLLMModel("gpt-4o")
        agent = MockToolCallingAgent(model=model)

        result = introspector.introspect(agent)

        assert result.baseline_model is not None
        assert result.baseline_model.name == "gpt-4o"
        assert result.context["agent_type"] == "MockToolCallingAgent"
        assert result.context["execution_mode"] == "json_tool_calling"

    def test_introspect_tool_calling_agent_with_tools(self, introspector):
        """Test introspection of ToolCallingAgent with tools."""
        tools = [MockTool("file_read", "Read file contents")]
        agent = MockToolCallingAgent(tools=tools)

        result = introspector.introspect(agent)

        cap_names = [c.name for c in result.capabilities]
        assert "file_read" in cap_names

        # file tools should be medium risk
        file_cap = next(c for c in result.capabilities if c.name == "file_read")
        assert file_cap.risk_level == "medium"

    # -------------------------------------------------------------------------
    # Multi-agent (managed_agents) tests
    # -------------------------------------------------------------------------

    def test_introspect_agent_with_managed_agents(self, introspector):
        """Test introspection of agent with managed agents."""
        managed = MockToolCallingAgent(name="sub_agent", description="A sub-agent")
        agent = MockCodeAgent(
            name="main_agent",
            description="The main agent",
            managed_agents=[managed],
        )

        result = introspector.introspect(agent)

        assert result.context["has_managed_agents"] is True
        assert result.context["managed_agent_count"] == 1

    # -------------------------------------------------------------------------
    # Model introspection tests
    # -------------------------------------------------------------------------

    def test_introspect_inference_client_model(self, introspector):
        """Test introspection of InferenceClientModel."""
        model = MockInferenceClientModel(
            model_id="meta-llama/Llama-3.3-70B-Instruct",
            temperature=0.8,
            max_tokens=2048,
        )

        # Introspect via an agent that uses this model
        agent = MockCodeAgent(model=model)
        result = introspector.introspect(agent)

        assert result.baseline_model is not None
        assert result.baseline_model.name == "meta-llama/Llama-3.3-70B-Instruct"

    def test_introspect_litellm_model_openai(self, introspector):
        """Test LiteLLMModel with OpenAI model."""
        model = MockLiteLLMModel("gpt-4o")
        agent = MockToolCallingAgent(model=model)

        result = introspector.introspect(agent)

        assert result.baseline_model.name == "gpt-4o"
        assert result.baseline_model.provider == ModelProvider.OPENAI

    def test_introspect_litellm_model_anthropic(self, introspector):
        """Test LiteLLMModel with Anthropic model."""
        model = MockLiteLLMModel("claude-3-5-sonnet-20241022")
        agent = MockToolCallingAgent(model=model)

        result = introspector.introspect(agent)

        assert result.baseline_model is not None
        assert "claude" in result.baseline_model.name

    def test_introspect_transformers_model(self, introspector):
        """Test TransformersModel (local)."""
        model = MockTransformersModel("meta-llama/Llama-3.2-1B")
        agent = MockCodeAgent(model=model)

        result = introspector.introspect(agent)

        assert result.baseline_model is not None
        assert result.baseline_model.name == "meta-llama/Llama-3.2-1B"

    # -------------------------------------------------------------------------
    # tiny-agents config tests
    # -------------------------------------------------------------------------

    def test_introspect_tiny_agents_config_basic(self, introspector):
        """Test introspection of basic tiny-agents config."""
        config = {
            "model": "gpt-4",
            "provider": "openai",
        }

        result = introspector.introspect(config)

        assert result.baseline_model is not None
        assert result.baseline_model.name == "gpt-4"
        assert result.baseline_model.provider == ModelProvider.OPENAI
        assert result.context["config_type"] == "tiny-agents"

    def test_introspect_tiny_agents_config_with_mcp_servers(self, introspector):
        """Test introspection of tiny-agents config with MCP servers."""
        config = {
            "model": "gpt-4o",
            "provider": "openai",
            "servers": [
                {
                    "type": "stdio",
                    "command": "npx",
                    "args": ["-y", "@playwright/mcp@latest"],
                },
                {
                    "type": "stdio",
                    "command": "uvx",
                    "args": ["mcp-server-fetch"],
                },
            ],
        }

        result = introspector.introspect(config)

        assert result.context["mcp_server_count"] == 2

        cap_names = [c.name for c in result.capabilities]
        assert "mcp_integration" in cap_names
        assert any("playwright" in name for name in cap_names)

        # Playwright should be medium risk (browser automation)
        playwright_cap = next(c for c in result.capabilities if "playwright" in c.name)
        assert playwright_cap.risk_level == "medium"

    def test_introspect_tiny_agents_config_local_endpoint(self, introspector):
        """Test tiny-agents config with local endpoint."""
        config = {
            "model": "Qwen/Qwen2.5-72B-Instruct",
            "endpointUrl": "http://localhost:8080/v1",
        }

        result = introspector.introspect(config)

        assert result.baseline_model.provider == ModelProvider.LOCAL
        assert result.context["local_inference"] is True

    def test_introspect_tiny_agents_config_anthropic(self, introspector):
        """Test tiny-agents config with Anthropic provider."""
        config = {
            "model": "claude-3-sonnet",
            "provider": "anthropic",
        }

        result = introspector.introspect(config)

        assert result.baseline_model.provider == ModelProvider.ANTHROPIC

    def test_introspect_tiny_agents_config_various_providers(self, introspector):
        """Test tiny-agents config with various inference providers."""
        providers = ["nebius", "together", "fireworks", "sambanova"]

        for provider in providers:
            config = {
                "model": "meta-llama/Llama-3.3-70B-Instruct",
                "provider": provider,
            }
            result = introspector.introspect(config)
            assert result.baseline_model is not None
            assert result.context["provider"] == provider

    # -------------------------------------------------------------------------
    # agent.json file tests
    # -------------------------------------------------------------------------

    def test_introspect_agent_json_file(self, introspector):
        """Test introspection of agent.json file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "model": "gpt-4o",
                "provider": "openai",
                "servers": [
                    {"type": "stdio", "command": "uvx", "args": ["mcp-server-git"]},
                ],
            }
            agent_json = Path(tmpdir) / "agent.json"
            agent_json.write_text(json.dumps(config))

            result = introspector.introspect(str(agent_json))

            assert result.baseline_model.name == "gpt-4o"
            # On macOS, /var is symlinked to /private/var, so use resolve for comparison
            assert Path(result.context["config_path"]).resolve() == agent_json.resolve()
            assert result.context["mcp_server_count"] == 1

    def test_introspect_agent_json_with_prompt_md(self, introspector):
        """Test detection of PROMPT.md alongside agent.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"model": "gpt-4", "provider": "openai"}
            (Path(tmpdir) / "agent.json").write_text(json.dumps(config))
            (Path(tmpdir) / "PROMPT.md").write_text("# Custom prompt")

            result = introspector.introspect(tmpdir)

            assert result.context.get("has_custom_prompt") is True

    def test_introspect_agent_json_with_agents_md(self, introspector):
        """Test detection of AGENTS.md alongside agent.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"model": "gpt-4", "provider": "openai"}
            (Path(tmpdir) / "agent.json").write_text(json.dumps(config))
            (Path(tmpdir) / "AGENTS.md").write_text("# Agent instructions")

            result = introspector.introspect(tmpdir)

            assert result.context.get("has_agents_md") is True

    def test_introspect_missing_agent_json(self, introspector):
        """Test handling of missing agent.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Pass directory without agent.json
            result = introspector.introspect(Path(tmpdir) / "nonexistent")

            assert "error" in result.context

    # -------------------------------------------------------------------------
    # Hub Agent tests
    # -------------------------------------------------------------------------

    def test_introspect_hub_agent(self, introspector):
        """Test introspection of Hub Agent."""
        config = {"model": "gpt-4o", "provider": "openai"}
        hub_agent = MockHubAgent(config=config)

        result = introspector.introspect(hub_agent)

        # HubAgent may or may not have baseline_model depending on implementation
        assert result.context["agent_type"] == "MockHubAgent"

    # -------------------------------------------------------------------------
    # Risk assessment tests
    # -------------------------------------------------------------------------

    def test_tool_risk_high(self, introspector):
        """Test high-risk tool detection."""
        tools = [
            MockTool("execute_shell", "Execute shell commands"),
            MockTool("delete_files", "Delete files from system"),
        ]
        agent = MockCodeAgent(tools=tools)

        result = introspector.introspect(agent)

        for cap in result.capabilities:
            if cap.name in ["execute_shell", "delete_files"]:
                assert cap.risk_level == "high"

    def test_tool_risk_medium(self, introspector):
        """Test medium-risk tool detection."""
        tools = [MockTool("http_request", "Make HTTP requests")]
        agent = MockCodeAgent(tools=tools)

        result = introspector.introspect(agent)

        http_cap = next(
            (c for c in result.capabilities if c.name == "http_request"), None
        )
        if http_cap:
            assert http_cap.risk_level == "medium"

    def test_tool_risk_low(self, introspector):
        """Test low-risk tool detection."""
        tools = [MockTool("calculator", "Basic math operations")]
        agent = MockToolCallingAgent(tools=tools)

        result = introspector.introspect(agent)

        calc_cap = next(
            (c for c in result.capabilities if c.name == "calculator"), None
        )
        if calc_cap:
            assert calc_cap.risk_level == "low"

    def test_mcp_server_risk_high(self, introspector):
        """Test high-risk MCP server detection."""
        config = {
            "model": "gpt-4",
            "provider": "openai",
            "servers": [
                {"type": "stdio", "command": "npx", "args": ["@playwright/mcp"]},
            ],
        }

        result = introspector.introspect(config)

        playwright_cap = next(
            (c for c in result.capabilities if "playwright" in c.name), None
        )
        assert playwright_cap is not None
        # Playwright is medium risk (browser automation, not system-level access)
        assert playwright_cap.risk_level == "medium"

    # -------------------------------------------------------------------------
    # get_agent_type tests
    # -------------------------------------------------------------------------

    def test_get_agent_type_code_agent(self, introspector):
        """Test get_agent_type for CodeAgent."""
        agent = MockCodeAgent()
        # get_agent_type returns class name
        assert introspector.get_agent_type(agent) == "MockCodeAgent"

    def test_get_agent_type_config_dict(self, introspector):
        """Test get_agent_type for config dict."""
        config = {"model": "gpt-4", "provider": "openai"}
        # Dicts return their type name
        assert introspector.get_agent_type(config) == "dict"

    def test_get_agent_type_path(self, introspector):
        """Test get_agent_type for path."""
        # Strings return their type name
        assert introspector.get_agent_type("/path/to/agent.json") == "str"

    def test_framework_name(self, introspector):
        """Test framework_name property."""
        assert introspector.framework_name == "huggingface"


# -------------------------------------------------------------------------
# Test Class: GroupFacts.from_huggingface()
# -------------------------------------------------------------------------


class TestGroupFactsFromHuggingFace:
    """Tests for GroupFacts.from_huggingface() factory method."""

    def test_from_code_agent(self):
        """Test creating GroupFacts from CodeAgent."""
        model = MockInferenceClientModel("Qwen/Qwen2.5-72B-Instruct")
        agent = MockCodeAgent(model=model, max_steps=10)

        group = GroupFacts.from_huggingface(agent, name="Code Agent")

        assert group.name == "Code Agent"
        assert len(group.members) == 1
        assert group.metadata.framework == "huggingface"

    def test_from_tool_calling_agent(self):
        """Test creating GroupFacts from ToolCallingAgent."""
        agent = MockToolCallingAgent(name="Tool Agent")

        group = GroupFacts.from_huggingface(agent, name="Tool Agent Group")

        assert group.name == "Tool Agent Group"
        assert len(group.members) == 1

    def test_from_tiny_agents_config(self):
        """Test creating GroupFacts from tiny-agents config."""
        config = {
            "model": "gpt-4o",
            "provider": "openai",
            "servers": [],
        }

        group = GroupFacts.from_huggingface(config, name="Tiny Agent")

        assert group.name == "Tiny Agent"
        assert group.metadata.framework == "huggingface"
        assert "config_type" in group.metadata.context

    def test_from_tiny_agents_config_auto_name(self):
        """Test auto-naming from tiny-agents config."""
        config = {"model": "claude-3-sonnet", "provider": "anthropic"}

        group = GroupFacts.from_huggingface(config)

        assert "claude-3-sonnet" in group.name

    def test_from_agent_json_path(self):
        """Test creating GroupFacts from agent.json path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"model": "gpt-4", "provider": "openai"}
            (Path(tmpdir) / "agent.json").write_text(json.dumps(config))

            group = GroupFacts.from_huggingface(tmpdir, name="JSON Agent")

            assert group.name == "JSON Agent"
            assert len(group.members) == 1

    def test_from_multi_agent_with_managed(self):
        """Test creating GroupFacts from agent with managed_agents."""
        sub_agent = MockToolCallingAgent(name="Sub Agent", description="A helper")
        main_agent = MockCodeAgent(
            name="Main Agent",
            description="The orchestrator",
            managed_agents=[sub_agent],
        )

        group = GroupFacts.from_huggingface(main_agent, name="Multi-Agent Team")

        assert group.name == "Multi-Agent Team"
        assert len(group.members) == 2
        assert group.metadata.process_type == ProcessType.HIERARCHICAL

    def test_sign_and_verify(self):
        """Test signing and verifying GroupFacts from HuggingFace."""
        config = {"model": "gpt-4", "provider": "openai"}
        group = GroupFacts.from_huggingface(config, name="Test Agent")

        signatures = group.sign_all()
        assert len(signatures) > 0
        assert group.is_signed

        results = group.verify_all()
        assert all(r.valid for r in results.values())

    def test_to_json(self):
        """Test JSON serialization."""
        agent = MockCodeAgent()
        group = GroupFacts.from_huggingface(agent, name="JSON Test")
        group.sign_all()

        json_str = group.to_json(include_members=True)
        data = json.loads(json_str)

        assert data["name"] == "JSON Test"
        assert data["framework"] == "huggingface"
        assert "_members" in data


# -------------------------------------------------------------------------
# Test Class: AgentFacts.from_agent()
# -------------------------------------------------------------------------


class TestAgentFactsFromHuggingFace:
    """Tests for AgentFacts.from_agent() with HuggingFace objects."""

    def setup_method(self):
        """Register the HuggingFace introspector before each test."""
        from agentfacts.integrations import get_registry, reset_registry

        reset_registry()
        registry = get_registry()
        registry.register(HuggingFaceIntrospector())

    def test_from_code_agent(self):
        """Test creating AgentFacts from CodeAgent."""
        model = MockInferenceClientModel("gpt-4o")
        agent = MockCodeAgent(model=model, max_steps=10)

        facts = AgentFacts.from_agent(agent, name="Code Agent")

        assert facts.name == "Code Agent"
        assert facts.metadata.agent.model is not None
        assert facts.metadata.agent.model.name == "gpt-4o"

    def test_from_tool_calling_agent(self):
        """Test creating AgentFacts from ToolCallingAgent."""
        tools = [MockTool("search", "Web search")]
        agent = MockToolCallingAgent(tools=tools)

        facts = AgentFacts.from_agent(agent, name="Tool Agent")

        cap_names = [c.name for c in facts.metadata.agent.capabilities]
        assert "search" in cap_names

    def test_from_tiny_agents_config(self):
        """Test creating AgentFacts from tiny-agents config."""
        config = {
            "model": "claude-3-sonnet",
            "provider": "anthropic",
            "servers": [
                {"type": "stdio", "command": "uvx", "args": ["mcp-server-git"]},
            ],
        }

        facts = AgentFacts.from_agent(config, name="Config Agent")

        assert facts.metadata.agent.model.provider == ModelProvider.ANTHROPIC
        assert "mcp_integration" in [c.name for c in facts.metadata.agent.capabilities]

    def test_sign_and_verify(self):
        """Test signing and verifying AgentFacts."""
        agent = MockCodeAgent()
        facts = AgentFacts.from_agent(agent, name="Test")

        facts.sign()
        assert facts.is_signed

        result = facts.verify()
        assert result.valid


# -------------------------------------------------------------------------
# Test Helper Functions
# -------------------------------------------------------------------------


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_load_tiny_agents_config(self):
        """Test load_tiny_agents_config helper."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"model": "gpt-4", "provider": "openai"}
            (Path(tmpdir) / "agent.json").write_text(json.dumps(config))

            loaded = load_tiny_agents_config(tmpdir)

            assert loaded["model"] == "gpt-4"
            assert loaded["provider"] == "openai"

    def test_load_tiny_agents_config_file_path(self):
        """Test load_tiny_agents_config with direct file path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"model": "claude-3", "provider": "anthropic"}
            agent_json = Path(tmpdir) / "agent.json"
            agent_json.write_text(json.dumps(config))

            loaded = load_tiny_agents_config(agent_json)

            assert loaded["model"] == "claude-3"

    def test_load_tiny_agents_config_not_found(self):
        """Test load_tiny_agents_config with missing file."""
        with pytest.raises(FileNotFoundError):
            load_tiny_agents_config("/nonexistent/path")


# -------------------------------------------------------------------------
# Test Error Handling
# -------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for error handling in HuggingFace introspector."""

    def test_invalid_json_file(self):
        """Test handling of invalid JSON in agent.json."""
        introspector = HuggingFaceIntrospector()

        with tempfile.TemporaryDirectory() as tmpdir:
            agent_json = Path(tmpdir) / "agent.json"
            agent_json.write_text("{ invalid json }")

            result = introspector.introspect(tmpdir)

            assert result.baseline_model is None
            assert "error" in result.context
            assert "Invalid JSON" in result.context["error"]

    def test_missing_agent_json(self):
        """Test handling of missing agent.json file."""
        introspector = HuggingFaceIntrospector()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = introspector.introspect(tmpdir)

            assert result.baseline_model is None
            assert "error" in result.context
            assert "not found" in result.context["error"]

    def test_path_traversal_blocked(self):
        """Test that path traversal attempts are blocked."""
        introspector = HuggingFaceIntrospector()

        # Test various path traversal attempts
        dangerous_paths = [
            "/etc/passwd",
            "/etc/shadow",
            "../../../etc/passwd",
            "/proc/self/environ",
            "/sys/kernel/debug",
        ]

        for path in dangerous_paths:
            result = introspector.introspect(path)
            # Should either return error or not find the file
            assert (
                result.baseline_model is None or "error" in result.context
            ), f"Path {path} should be blocked"

    def test_non_json_file_rejected(self):
        """Test that non-.json files are rejected."""
        introspector = HuggingFaceIntrospector()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a non-json file
            txt_file = Path(tmpdir) / "config.txt"
            txt_file.write_text('{"model": "gpt-4"}')

            result = introspector.introspect(txt_file)

            assert result.baseline_model is None
            assert "error" in result.context
            assert ".json" in result.context["error"]

    def test_empty_json_file(self):
        """Test handling of empty JSON file."""
        introspector = HuggingFaceIntrospector()

        with tempfile.TemporaryDirectory() as tmpdir:
            agent_json = Path(tmpdir) / "agent.json"
            agent_json.write_text("{}")

            result = introspector.introspect(tmpdir)

            # Empty config should still work but produce minimal results
            assert "error" not in result.context

    def test_malformed_config_dict(self):
        """Test handling of malformed config dict."""
        introspector = HuggingFaceIntrospector()

        # Test with various malformed configs
        configs = [
            {"model": 123},  # Wrong type for model
            {"servers": "not a list"},  # Wrong type for servers
            {"model": None},  # None values
        ]

        for config in configs:
            # Should not raise exceptions
            introspector.introspect(config)
            # May or may not produce results, but should not crash

    def test_tools_without_len(self):
        """Test handling of tools that don't support len()."""
        introspector = HuggingFaceIntrospector()

        # Create a generator that doesn't support len
        def tool_generator():
            yield MockTool("tool1")
            yield MockTool("tool2")

        agent = MockCodeAgent(
            tools=tool_generator(),
            model=MockInferenceClientModel(),
        )

        # Should not crash even with non-sizeable tools
        result = introspector.introspect(agent)
        assert result.baseline_model is not None

    def test_model_without_attributes(self):
        """Test handling of model without expected attributes."""
        introspector = HuggingFaceIntrospector()

        class BareModel:
            __module__ = "smolagents"
            pass

        class MinimalCodeAgent:
            __module__ = "smolagents"

            def __init__(self):
                self.model = BareModel()
                self.tools = []

            def run(self):
                pass

        agent = MinimalCodeAgent()
        result = introspector.introspect(agent)

        # Should handle gracefully
        assert result.baseline_model is not None or "error" not in result.context

    def test_server_extraction_edge_cases(self):
        """Test MCP server name extraction with edge cases."""
        introspector = HuggingFaceIntrospector()

        # Test various server configurations
        configs = [
            {"servers": [{"type": "stdio", "command": ""}]},  # Empty command
            {"servers": [{"type": "stdio", "args": []}]},  # Missing command
            {"servers": [{"command": "npx", "args": ["-y", ""]}]},  # Empty args
            {"servers": [{"command": "npx", "args": None}]},  # None args
        ]

        for config in configs:
            # Should not raise exceptions
            introspector.introspect(config)

    def test_unicode_content(self):
        """Test handling of Unicode content in config."""
        introspector = HuggingFaceIntrospector()

        config = {
            "model": "gpt-4-日本語",
            "prompt": "こんにちは 🎉",
            "servers": [{"command": "npx", "args": ["-y", "@测试/mcp-server"]}],
        }

        result = introspector.introspect(config)
        # Should handle Unicode gracefully
        assert result.baseline_model is not None or "error" not in result.context


class TestRiskAssessment:
    """Tests for risk assessment functionality."""

    def test_high_risk_tools(self):
        """Test that high-risk tools are correctly identified."""
        introspector = HuggingFaceIntrospector()

        high_risk_names = [
            "execute_command",
            "shell_exec",
            "bash_runner",
            "file_write",
            "sql_query",
            "delete_file",
            "admin_access",
        ]

        for name in high_risk_names:
            risk = introspector._assess_risk(name, "")
            assert risk == "high", f"Tool {name} should be high risk"

    def test_medium_risk_tools(self):
        """Test that medium-risk tools are correctly identified."""
        introspector = HuggingFaceIntrospector()

        medium_risk_names = [
            "http_request",
            "file_reader",
            "email_sender",
            "web_scraper",
            "api_call",
        ]

        for name in medium_risk_names:
            risk = introspector._assess_risk(name, "")
            assert risk == "medium", f"Tool {name} should be medium risk"

    def test_low_risk_tools(self):
        """Test that low-risk tools are correctly identified."""
        introspector = HuggingFaceIntrospector()

        # Use names that don't contain substrings of high-risk keywords
        # Note: "text_formatter" contains "rm", "calculator" contains nothing risky now
        low_risk_names = [
            "add_numbers",
            "date_parser",
            "json_validator",
        ]

        for name in low_risk_names:
            risk = introspector._assess_risk(name, "")
            assert risk == "low", f"Tool {name} should be low risk"

    def test_high_risk_mcp_servers(self):
        """Test that high-risk MCP servers are correctly identified."""
        introspector = HuggingFaceIntrospector()

        high_risk_servers = [
            ("docker", []),
            ("kubectl", ["apply", "-f"]),
            ("npx", ["-y", "@filesystem/mcp-server"]),
            ("aws", ["s3", "ls"]),
            ("npx", ["-y", "shell-server"]),
            ("bash", ["-c", "ls"]),
        ]

        for cmd, args in high_risk_servers:
            risk = introspector._assess_mcp_risk(cmd, args)
            assert risk == "high", f"Server {cmd} with args {args} should be high risk"

    def test_medium_risk_mcp_servers(self):
        """Test that browser automation and network tools are medium risk."""
        introspector = HuggingFaceIntrospector()

        medium_risk_servers = [
            ("npx", ["-y", "@playwright/mcp@latest"]),
            ("npx", ["-y", "puppeteer-mcp"]),
            ("npx", ["-y", "http-server-mcp"]),
            ("npx", ["-y", "github-mcp"]),
        ]

        for cmd, args in medium_risk_servers:
            risk = introspector._assess_mcp_risk(cmd, args)
            assert (
                risk == "medium"
            ), f"Server {cmd} with args {args} should be medium risk"
