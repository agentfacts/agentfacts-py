"""
Tests for the integration registry and framework detection.
"""

import pytest

from agentfacts.core import AgentFacts
from agentfacts.integrations import (
    BaseIntrospector,
    IntegrationRegistry,
    IntrospectionResult,
    get_registry,
    reset_registry,
)
from agentfacts.models import (
    BaselineModel,
    Capability,
    ModelProvider,
    OperationalConstraints,
)


class MockAgent:
    """Mock agent for testing."""

    pass


class MockLangChainAgent:
    """Mock LangChain-like agent."""

    __module__ = "langchain.agents"

    def __init__(self):
        self.tools = []
        self.agent = None

    def invoke(self, x):
        return x

    def batch(self, x):
        return x


class MockCrewAIAgent:
    """Mock CrewAI-like agent."""

    __module__ = "crewai.agents"

    def __init__(self):
        self.role = "Researcher"
        self.goal = "Find info"


class MockAutoGenAgent:
    """Mock AutoGen-like agent."""

    __module__ = "autogen.agentchat"

    def __init__(self):
        self.name = "assistant"
        self.description = "A helpful assistant"


class TestIntrospector(BaseIntrospector):
    """Test introspector implementation."""

    @property
    def framework_name(self) -> str:
        return "test_framework"

    def can_introspect(self, obj) -> bool:
        return isinstance(obj, MockAgent)

    def introspect(self, obj) -> IntrospectionResult:
        return IntrospectionResult(
            framework="test_framework",
            baseline_model=BaselineModel(
                name="test-model", provider=ModelProvider.UNKNOWN
            ),
            capabilities=[Capability(name="test_tool")],
            constraints=OperationalConstraints(),
            context={"test_key": "test_value"},
        )


class TestBaseIntrospector:
    """Tests for BaseIntrospector."""

    def test_abstract_methods(self):
        """Test that abstract methods must be implemented."""
        with pytest.raises(TypeError):
            # Can't instantiate abstract class
            BaseIntrospector()

    def test_concrete_implementation(self):
        """Test concrete implementation works."""
        introspector = TestIntrospector()
        assert introspector.framework_name == "test_framework"
        assert introspector.can_introspect(MockAgent())
        assert not introspector.can_introspect("not an agent")

    def test_get_agent_type_default(self):
        """Test default agent type is class name."""
        introspector = TestIntrospector()
        assert introspector.get_agent_type(MockAgent()) == "MockAgent"


class TestIntrospectionResult:
    """Tests for IntrospectionResult."""

    def test_create_result(self):
        """Test creating introspection result."""
        result = IntrospectionResult(
            framework="test",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
            capabilities=[Capability(name="tool1")],
            constraints=OperationalConstraints(),
            context={"key": "value"},
        )
        assert result.framework == "test"
        assert result.baseline_model.name == "gpt-4"
        assert len(result.capabilities) == 1

    def test_result_repr(self):
        """Test result string representation."""
        result = IntrospectionResult(
            framework="langchain",
            baseline_model=BaselineModel(name="gpt-4", provider=ModelProvider.OPENAI),
            capabilities=[Capability(name="a"), Capability(name="b")],
            constraints=OperationalConstraints(),
            context={},
        )
        repr_str = repr(result)
        assert "langchain" in repr_str
        assert "gpt-4" in repr_str
        assert "2" in repr_str  # capabilities count


class TestIntegrationRegistry:
    """Tests for IntegrationRegistry."""

    def test_create_empty_registry(self):
        """Test creating empty registry."""
        registry = IntegrationRegistry()
        assert registry.available_frameworks == []

    def test_register_introspector(self):
        """Test registering an introspector."""
        registry = IntegrationRegistry()
        introspector = TestIntrospector()

        registry.register(introspector)

        assert "test_framework" in registry.available_frameworks
        assert registry.has_framework("test_framework")

    def test_unregister_introspector(self):
        """Test unregistering an introspector."""
        registry = IntegrationRegistry()
        registry.register(TestIntrospector())

        result = registry.unregister("test_framework")

        assert result is True
        assert "test_framework" not in registry.available_frameworks

    def test_unregister_nonexistent(self):
        """Test unregistering non-existent framework."""
        registry = IntegrationRegistry()
        result = registry.unregister("nonexistent")
        assert result is False

    def test_detect_framework_by_introspector(self):
        """Test framework detection via introspector."""
        registry = IntegrationRegistry()
        registry.register(TestIntrospector())

        framework = registry.detect_framework(MockAgent())
        assert framework == "test_framework"

    def test_detect_framework_by_module_langchain(self):
        """Test framework detection via module name for LangChain."""
        registry = IntegrationRegistry()
        framework = registry.detect_framework(MockLangChainAgent())
        assert framework == "langchain"

    def test_detect_framework_by_module_crewai(self):
        """Test framework detection via module name for CrewAI."""
        registry = IntegrationRegistry()
        framework = registry.detect_framework(MockCrewAIAgent())
        assert framework == "crewai"

    def test_detect_framework_by_module_autogen(self):
        """Test framework detection via module name for AutoGen."""
        registry = IntegrationRegistry()
        framework = registry.detect_framework(MockAutoGenAgent())
        assert framework == "autogen"

    def test_detect_framework_unknown(self):
        """Test framework detection returns None for unknown."""
        registry = IntegrationRegistry()
        framework = registry.detect_framework("unknown object")
        assert framework is None

    def test_introspect_with_registered(self):
        """Test introspection with registered introspector."""
        registry = IntegrationRegistry()
        registry.register(TestIntrospector())

        result = registry.introspect(MockAgent())

        assert result.framework == "test_framework"
        assert result.baseline_model.name == "test-model"
        assert result.context["test_key"] == "test_value"

    def test_introspect_explicit_framework(self):
        """Test introspection with explicit framework."""
        registry = IntegrationRegistry()
        registry.register(TestIntrospector())

        # Even though MockAgent would be detected, we can force framework
        result = registry.introspect(MockAgent(), framework="test_framework")
        assert result.framework == "test_framework"

    def test_introspect_unknown_framework_raises(self):
        """Test introspection raises for unknown framework."""
        registry = IntegrationRegistry()

        with pytest.raises(ValueError, match="Cannot detect framework"):
            registry.introspect("unknown")

    def test_introspect_unregistered_framework_raises(self):
        """Test introspection raises for unregistered framework."""
        registry = IntegrationRegistry()

        with pytest.raises(ValueError, match="No integration registered"):
            registry.introspect(MockCrewAIAgent(), framework="crewai")

    def test_introspect_safe_success(self):
        """Test safe introspection on success."""
        registry = IntegrationRegistry()
        registry.register(TestIntrospector())

        result = registry.introspect_safe(MockAgent())
        assert result is not None
        assert result.framework == "test_framework"

    def test_introspect_safe_failure(self):
        """Test safe introspection returns None on failure."""
        registry = IntegrationRegistry()
        result = registry.introspect_safe("unknown")
        assert result is None


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def test_get_registry_creates_singleton(self):
        """Test get_registry creates singleton."""
        reset_registry()
        registry1 = get_registry()
        registry2 = get_registry()
        assert registry1 is registry2

    def test_reset_registry(self):
        """Test reset_registry clears singleton."""
        registry1 = get_registry()
        reset_registry()
        registry2 = get_registry()
        assert registry1 is not registry2

    def test_global_registry_has_langchain(self):
        """Test global registry registers LangChain by default."""
        reset_registry()
        registry = get_registry()
        # LangChain introspector should be registered if langchain is installed
        # This might be True or False depending on environment
        # Just verify it doesn't error
        assert isinstance(registry.available_frameworks, list)

    def test_global_registry_registers_langchain(self):
        """Test LangChain integration registers in the global registry."""
        reset_registry()
        registry = get_registry()
        assert registry.has_framework("langchain")


class TestLangChainIntrospector:
    """Tests for LangChain introspector (if available)."""

    def test_langchain_introspector_exists(self):
        """Test LangChain introspector can be imported."""
        try:
            from agentfacts.integrations.langchain.introspector import (
                LangChainIntegration as LangChainIntrospector,
            )

            introspector = LangChainIntrospector()
            assert introspector.framework_name == "langchain"
        except ImportError:
            pytest.skip("LangChain not installed")

    def test_langchain_can_introspect_executor(self):
        """Test LangChain introspector detects executor-like objects."""
        try:
            from agentfacts.integrations.langchain.introspector import (
                LangChainIntegration as LangChainIntrospector,
            )

            introspector = LangChainIntrospector()

            # Mock executor-like object
            class FakeExecutor:
                __module__ = "langchain.agents"

                def __init__(self):
                    self.agent = None
                    self.tools = []

            assert introspector.can_introspect(FakeExecutor())
        except ImportError:
            pytest.skip("LangChain not installed")

    def test_langchain_cannot_introspect_random(self):
        """Test LangChain introspector rejects random objects."""
        try:
            from agentfacts.integrations.langchain.introspector import (
                LangChainIntegration as LangChainIntrospector,
            )

            introspector = LangChainIntrospector()
            assert not introspector.can_introspect("random string")
            assert not introspector.can_introspect(123)
        except ImportError:
            pytest.skip("LangChain not installed")

    def test_get_callback_handler_import(self):
        """Test callback handler import path resolves correctly."""
        facts = AgentFacts(name="Callback Test")
        try:
            import langchain_core  # noqa: F401
        except ImportError:
            with pytest.raises(ImportError) as excinfo:
                facts.get_callback_handler()
            assert "LangChain is required" in str(excinfo.value)
        else:
            handler = facts.get_callback_handler()
            assert handler is not None
