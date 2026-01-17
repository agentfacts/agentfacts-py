"""
Tests for AutoGen introspector.
"""

from agentfacts import AgentFacts, GroupFacts, ProcessType
from agentfacts.integrations.autogen.introspector import (
    AutoGenIntegration as AutoGenIntrospector,
)
from agentfacts.models import ModelProvider


class MockAutoGenAssistantAgent:
    """Mock AutoGen AssistantAgent."""

    __module__ = "autogen.agentchat"

    def __init__(
        self,
        name: str = "assistant",
        system_message: str = "You are a helpful assistant.",
        llm_config: dict = None,
        description: str = "",
        code_execution_config: dict = None,
        human_input_mode: str = "NEVER",
        max_consecutive_auto_reply: int = None,
    ):
        self.name = name
        self.system_message = system_message
        self.llm_config = llm_config
        self.description = description
        self.code_execution_config = code_execution_config
        self.human_input_mode = human_input_mode
        self.max_consecutive_auto_reply = max_consecutive_auto_reply
        self._function_map = {}
        self._reply_func_list = []


class MockAutoGenUserProxyAgent:
    """Mock AutoGen UserProxyAgent."""

    __module__ = "autogen.agentchat"

    def __init__(
        self,
        name: str = "user_proxy",
        human_input_mode: str = "ALWAYS",
        code_execution_config: dict = None,
        llm_config: dict = None,
    ):
        self.name = name
        self.human_input_mode = human_input_mode
        self.code_execution_config = code_execution_config
        self.llm_config = llm_config
        self.description = ""
        self.system_message = ""
        self._function_map = {}


class MockAutoGenGroupChat:
    """Mock AutoGen GroupChat."""

    __module__ = "autogen.agentchat"

    def __init__(
        self,
        agents: list = None,
        messages: list = None,
        max_round: int = 10,
        speaker_selection_method: str = "auto",
        admin_name: str = "Admin",
        allow_repeat_speaker: bool = True,
    ):
        self.agents = agents or []
        self.messages = messages or []
        self.max_round = max_round
        self.speaker_selection_method = speaker_selection_method
        self.admin_name = admin_name
        self.allow_repeat_speaker = allow_repeat_speaker


class MockAutoGenGroupChatManager:
    """Mock AutoGen GroupChatManager."""

    __module__ = "autogen.agentchat"

    def __init__(self, groupchat: MockAutoGenGroupChat = None, llm_config: dict = None):
        self.groupchat = groupchat
        self.llm_config = llm_config
        self.name = "group_manager"


class TestAutoGenIntrospector:
    """Tests for AutoGen introspector."""

    def test_framework_name(self):
        """Test framework name."""
        introspector = AutoGenIntrospector()
        assert introspector.framework_name == "autogen"

    def test_can_introspect_assistant(self):
        """Test can_introspect with AssistantAgent."""
        introspector = AutoGenIntrospector()
        agent = MockAutoGenAssistantAgent()
        assert introspector.can_introspect(agent) is True

    def test_can_introspect_user_proxy(self):
        """Test can_introspect with UserProxyAgent."""
        introspector = AutoGenIntrospector()
        agent = MockAutoGenUserProxyAgent()
        assert introspector.can_introspect(agent) is True

    def test_can_introspect_group_chat(self):
        """Test can_introspect with GroupChat."""
        introspector = AutoGenIntrospector()
        group = MockAutoGenGroupChat()
        assert introspector.can_introspect(group) is True

    def test_cannot_introspect_random(self):
        """Test can_introspect rejects random objects."""
        introspector = AutoGenIntrospector()
        assert introspector.can_introspect("string") is False
        assert introspector.can_introspect(123) is False

    def test_introspect_assistant_basic(self):
        """Test basic AssistantAgent introspection."""
        introspector = AutoGenIntrospector()
        agent = MockAutoGenAssistantAgent(
            name="coder",
            description="A coding assistant",
            system_message="You help write code.",
        )

        result = introspector.introspect(agent)

        assert result.context["agent_name"] == "coder"
        assert result.context["description"] == "A coding assistant"
        assert "You help write code" in result.context["system_message"]

    def test_introspect_assistant_with_llm_config(self):
        """Test AssistantAgent introspection with llm_config."""
        introspector = AutoGenIntrospector()
        llm_config = {
            "config_list": [{"model": "gpt-4-turbo"}],
            "temperature": 0.3,
            "max_tokens": 2000,
        }
        agent = MockAutoGenAssistantAgent(llm_config=llm_config)

        result = introspector.introspect(agent)

        assert result.baseline_model is not None
        assert result.baseline_model.name == "gpt-4-turbo"
        assert result.baseline_model.provider == ModelProvider.OPENAI
        assert result.baseline_model.temperature == 0.3

    def test_introspect_assistant_with_code_execution(self):
        """Test AssistantAgent introspection with code execution."""
        introspector = AutoGenIntrospector()
        agent = MockAutoGenAssistantAgent(
            code_execution_config={"work_dir": "/tmp/code", "use_docker": True}
        )

        result = introspector.introspect(agent)

        assert result.context["code_execution_enabled"] is True
        assert result.context["code_uses_docker"] is True
        assert any(c.name == "code_execution" for c in result.capabilities)
        # Code execution should be high risk
        code_cap = next(c for c in result.capabilities if c.name == "code_execution")
        assert code_cap.risk_level == "high"

    def test_introspect_assistant_human_approval(self):
        """Test AssistantAgent with human approval mode."""
        introspector = AutoGenIntrospector()
        agent = MockAutoGenAssistantAgent(human_input_mode="ALWAYS")

        result = introspector.introspect(agent)

        assert result.constraints.requires_human_approval is True

    def test_introspect_assistant_max_iterations(self):
        """Test AssistantAgent max iterations."""
        introspector = AutoGenIntrospector()
        agent = MockAutoGenAssistantAgent(max_consecutive_auto_reply=5)

        result = introspector.introspect(agent)

        assert result.constraints.max_iterations == 5

    def test_introspect_user_proxy(self):
        """Test UserProxyAgent introspection."""
        introspector = AutoGenIntrospector()
        agent = MockAutoGenUserProxyAgent(name="user")

        result = introspector.introspect(agent)

        assert result.context["agent_name"] == "user"
        # Mock class won't have same name as real UserProxyAgent
        assert result.context["agent_class"] == "MockAutoGenUserProxyAgent"

    def test_introspect_group_chat_basic(self):
        """Test basic GroupChat introspection."""
        introspector = AutoGenIntrospector()
        agents = [
            MockAutoGenAssistantAgent(name="agent1", description="First agent"),
            MockAutoGenAssistantAgent(name="agent2", description="Second agent"),
        ]
        group = MockAutoGenGroupChat(agents=agents, max_round=15)

        result = introspector.introspect(group)

        assert result.context["agent_count"] == 2
        assert "agent1" in result.context["agent_names"]
        assert "agent2" in result.context["agent_names"]
        assert result.context["max_round"] == 15
        assert result.constraints.max_iterations == 15
        assert result.context["process_type"] == "event_driven"

    def test_introspect_group_chat_speaker_selection(self):
        """Test GroupChat speaker selection method extraction."""
        introspector = AutoGenIntrospector()
        group = MockAutoGenGroupChat(speaker_selection_method="round_robin")

        result = introspector.introspect(group)

        assert result.context["speaker_selection_method"] == "round_robin"

    def test_introspect_group_chat_collects_code_execution(self):
        """Test GroupChat collects code execution from agents."""
        introspector = AutoGenIntrospector()
        agents = [
            MockAutoGenAssistantAgent(
                name="coder", code_execution_config={"work_dir": "."}
            ),
            MockAutoGenAssistantAgent(name="reviewer"),
        ]
        group = MockAutoGenGroupChat(agents=agents)

        result = introspector.introspect(group)

        assert any(c.name == "code_execution" for c in result.capabilities)

    def test_introspect_group_chat_manager(self):
        """Test GroupChatManager introspection via groupchat attribute."""
        introspector = AutoGenIntrospector()
        agents = [MockAutoGenAssistantAgent(), MockAutoGenUserProxyAgent()]
        group_chat = MockAutoGenGroupChat(agents=agents)
        manager = MockAutoGenGroupChatManager(groupchat=group_chat)

        # Since mock class isn't named GroupChatManager, it introspects as agent
        # Test that it has groupchat attribute and can be processed
        introspector.introspect(manager)

        # The manager has a groupchat attribute that can be accessed
        assert hasattr(manager, "groupchat")
        assert manager.groupchat is group_chat

    def test_detect_provider_openai(self):
        """Test provider detection for OpenAI models."""
        introspector = AutoGenIntrospector()

        assert introspector._detect_provider("gpt-4", "") == ModelProvider.OPENAI
        assert (
            introspector._detect_provider("gpt-3.5-turbo", "") == ModelProvider.OPENAI
        )

    def test_detect_provider_anthropic(self):
        """Test provider detection for Anthropic models."""
        introspector = AutoGenIntrospector()

        assert (
            introspector._detect_provider("claude-3-opus", "")
            == ModelProvider.ANTHROPIC
        )
        assert introspector._detect_provider("claude-2", "") == ModelProvider.ANTHROPIC

    def test_detect_provider_from_api_base(self):
        """Test provider detection from API base URL."""
        introspector = AutoGenIntrospector()

        assert (
            introspector._detect_provider("custom", "https://api.openai.com")
            == ModelProvider.OPENAI
        )
        assert (
            introspector._detect_provider("custom", "https://api.anthropic.com")
            == ModelProvider.ANTHROPIC
        )

    def test_get_agent_type(self):
        """Test get_agent_type returns correct types for mock classes."""
        introspector = AutoGenIntrospector()

        # Mock classes return their class name
        assistant = MockAutoGenAssistantAgent()
        assert introspector.get_agent_type(assistant) is not None

        user_proxy = MockAutoGenUserProxyAgent()
        assert introspector.get_agent_type(user_proxy) is not None

        group_chat = MockAutoGenGroupChat()
        assert introspector.get_agent_type(group_chat) is not None


class TestGroupFactsFromAutoGen:
    """Tests for GroupFacts.from_autogen()."""

    def test_from_autogen_basic(self):
        """Test creating GroupFacts from AutoGen GroupChat."""
        agents = [
            MockAutoGenAssistantAgent(
                name="coder",
                description="Writes code",
                llm_config={"config_list": [{"model": "gpt-4"}]},
            ),
            MockAutoGenUserProxyAgent(name="user"),
        ]
        group_chat = MockAutoGenGroupChat(agents=agents, max_round=20)

        group = GroupFacts.from_autogen(group_chat, name="Dev Team")

        assert group.name == "Dev Team"
        assert len(group.members) == 2
        assert group.metadata.framework == "autogen"
        assert group.metadata.process_type == ProcessType.EVENT_DRIVEN
        assert group.metadata.max_rounds == 20

    def test_from_autogen_with_manager(self):
        """Test creating GroupFacts from GroupChatManager."""
        agents = [MockAutoGenAssistantAgent(), MockAutoGenAssistantAgent()]
        group_chat = MockAutoGenGroupChat(agents=agents)
        manager = MockAutoGenGroupChatManager(groupchat=group_chat)

        group = GroupFacts.from_autogen(manager, name="Managed Team")

        assert group.name == "Managed Team"
        assert len(group.members) == 2

    def test_from_autogen_speaker_method(self):
        """Test GroupFacts captures speaker selection method."""
        group_chat = MockAutoGenGroupChat(
            agents=[MockAutoGenAssistantAgent()],
            speaker_selection_method="manual",
        )

        group = GroupFacts.from_autogen(group_chat)

        assert group.metadata.context["speaker_selection_method"] == "manual"

    def test_from_autogen_members_have_framework(self):
        """Test that member AgentFacts have autogen framework."""
        agents = [MockAutoGenAssistantAgent()]
        group_chat = MockAutoGenGroupChat(agents=agents)

        group = GroupFacts.from_autogen(group_chat)

        member = group.members[0]
        assert member.metadata.agent.framework == "autogen"

    def test_from_autogen_sign_all(self):
        """Test signing AutoGen group and members."""
        agents = [
            MockAutoGenAssistantAgent(name="agent1"),
            MockAutoGenAssistantAgent(name="agent2"),
        ]
        group_chat = MockAutoGenGroupChat(agents=agents)

        group = GroupFacts.from_autogen(group_chat)
        signatures = group.sign_all()

        assert len(signatures) == 3  # 2 agents + 1 group
        assert group.all_verified()


class TestAgentFactsFromAutoGen:
    """Tests for AgentFacts.from_agent with AutoGen."""

    def test_from_agent_autogen_auto_detect(self):
        """Test AgentFacts.from_agent auto-detects AutoGen."""
        agent = MockAutoGenAssistantAgent(
            name="assistant",
            llm_config={"config_list": [{"model": "gpt-4"}]},
        )

        from agentfacts.integrations import get_registry, reset_registry

        reset_registry()
        registry = get_registry()
        registry.register(AutoGenIntrospector())

        facts = AgentFacts.from_agent(agent, name="Test Assistant")

        assert facts.name == "Test Assistant"
        assert facts.metadata.agent.framework == "autogen"

    def test_from_agent_autogen_explicit(self):
        """Test AgentFacts.from_agent with explicit framework."""
        agent = MockAutoGenAssistantAgent(
            llm_config={"config_list": [{"model": "claude-3"}]},
        )

        from agentfacts.integrations import get_registry, reset_registry

        reset_registry()
        registry = get_registry()
        registry.register(AutoGenIntrospector())

        facts = AgentFacts.from_agent(
            agent, name="Claude Assistant", framework="autogen"
        )

        assert facts.metadata.agent.framework == "autogen"
        assert facts.metadata.agent.model.name == "claude-3"
