"""
Testing utilities for AgentFacts SDK.

Provides mock objects and helpers for testing introspectors
and AgentFacts integrations without requiring actual framework
dependencies.

Example:
    ```python
    from agentfacts.testing import MockAgent, create_mock_tool, create_mock_llm

    # Create a mock agent for testing
    agent = MockAgent(
        module="crewai",
        name="TestAgent",
        tools=[
            create_mock_tool("search", "Search the web"),
            create_mock_tool("execute", "Run code", risk="high"),
        ],
        llm=create_mock_llm("gpt-4", "openai"),
    )

    # Test your introspector
    introspector = MyFrameworkIntrospector()
    assert introspector.can_introspect(agent)
    result = introspector.introspect(agent)
    assert result.capability_count == 2
    ```
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MockTool:
    """
    Mock tool for testing introspectors.

    Mimics the common attributes of tools across frameworks.

    Attributes:
        name: Tool name
        description: Tool description
        parameters: Tool parameter schema
        risk_hint: Expected risk level (for test assertions)
    """

    name: str = "mock_tool"
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    risk_hint: str = "low"

    # Additional attributes frameworks might look for
    func: Any | None = None
    args_schema: Any | None = None

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        """Make the tool callable for frameworks that invoke tools."""
        return f"MockTool({self.name}) called"


@dataclass
class MockLLM:
    """
    Mock LLM for testing introspectors.

    Mimics common LLM attributes across frameworks.

    Attributes:
        model_name: Model identifier (e.g., "gpt-4")
        provider_hint: Expected provider (for test assertions)
        temperature: Sampling temperature
        max_tokens: Maximum output tokens
    """

    model_name: str = "gpt-4"
    provider_hint: str = "openai"
    temperature: float = 0.7
    max_tokens: int = 4096

    # Alternative attribute names frameworks might use
    model: str = ""
    model_id: str = ""

    def __post_init__(self) -> None:
        """Set alternative attribute names."""
        if not self.model:
            self.model = self.model_name
        if not self.model_id:
            self.model_id = self.model_name


class MockAgent:
    """
    Configurable mock agent for testing introspectors.

    Creates a mock agent that appears to be from a specific framework
    by setting __module__ appropriately.

    Attributes:
        module: Framework module name (e.g., "crewai", "autogen")
        name: Agent name
        llm: Mock LLM instance
        tools: List of mock tools
        max_iterations: Maximum iterations constraint
        role: Role name (for CrewAI-style agents)
        goal: Agent goal
        backstory: Agent backstory
        memory: Whether agent has memory
        verbose: Verbose mode flag

    Example:
        ```python
        agent = MockAgent(
            module="crewai.agents",
            name="Researcher",
            role="Senior Researcher",
            goal="Find accurate information",
            llm=MockLLM("claude-3-opus", "anthropic"),
        )
        ```
    """

    def __init__(
        self,
        module: str = "mock_framework",
        name: str = "MockAgent",
        llm: MockLLM | None = None,
        tools: list[MockTool] | None = None,
        max_iterations: int = 10,
        role: str = "",
        goal: str = "",
        backstory: str = "",
        memory: bool = False,
        verbose: bool = False,
        **extra_attrs: Any,
    ):
        self.name = name
        self.llm = llm or MockLLM()
        self.tools = tools or []
        self.max_iterations = max_iterations
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.memory = memory
        self.verbose = verbose

        # Set any extra attributes
        for key, value in extra_attrs.items():
            setattr(self, key, value)

        # IMPORTANT: Set __module__ for framework detection
        # This makes can_introspect() work correctly
        self.__class__.__module__ = module


class MockCrew:
    """
    Mock CrewAI Crew for testing GroupFacts.

    Example:
        ```python
        crew = MockCrew(
            agents=[MockAgent(module="crewai"), MockAgent(module="crewai")],
            process="sequential",
        )
        group = GroupFacts.from_crewai(crew)
        ```
    """

    def __init__(
        self,
        agents: list[MockAgent] | None = None,
        tasks: list[Any] | None = None,
        process: str = "sequential",
        memory: bool = False,
        verbose: bool = False,
        **extra_attrs: Any,
    ):
        self.agents = agents or []
        self.tasks = tasks or []
        self.process = process
        self.memory = memory
        self.verbose = verbose

        for key, value in extra_attrs.items():
            setattr(self, key, value)

        self.__class__.__module__ = "crewai.crew"


class MockGroupChat:
    """
    Mock AutoGen GroupChat for testing GroupFacts.

    Example:
        ```python
        chat = MockGroupChat(
            agents=[MockAgent(module="autogen"), MockAgent(module="autogen")],
            max_round=10,
        )
        group = GroupFacts.from_autogen(chat)
        ```
    """

    def __init__(
        self,
        agents: list[MockAgent] | None = None,
        max_round: int = 10,
        admin_name: str = "Admin",
        **extra_attrs: Any,
    ):
        self.agents = agents or []
        self.max_round = max_round
        self.admin_name = admin_name

        for key, value in extra_attrs.items():
            setattr(self, key, value)

        self.__class__.__module__ = "autogen.agentchat"


def create_mock_tool(
    name: str,
    description: str = "",
    parameters: dict[str, Any] | None = None,
    risk: str = "low",
) -> MockTool:
    """
    Create a mock tool for testing.

    Args:
        name: Tool name
        description: Tool description
        parameters: Tool parameter schema
        risk: Expected risk level ("low", "medium", "high")

    Returns:
        MockTool instance

    Example:
        ```python
        tool = create_mock_tool("search", "Search the web", risk="medium")
        ```
    """
    return MockTool(
        name=name,
        description=description,
        parameters=parameters or {},
        risk_hint=risk,
    )


def create_mock_llm(
    model_name: str = "gpt-4",
    provider: str = "openai",
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> MockLLM:
    """
    Create a mock LLM for testing.

    Args:
        model_name: Model identifier
        provider: Expected provider
        temperature: Sampling temperature
        max_tokens: Maximum output tokens

    Returns:
        MockLLM instance

    Example:
        ```python
        llm = create_mock_llm("claude-3-opus", "anthropic")
        ```
    """
    return MockLLM(
        model_name=model_name,
        provider_hint=provider,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def create_mock_agent(
    framework: str = "langchain",
    name: str = "TestAgent",
    model_name: str = "gpt-4",
    tool_names: list[str] | None = None,
    **kwargs: Any,
) -> MockAgent:
    """
    Create a mock agent with common defaults.

    Convenience function that creates a MockAgent with an LLM
    and optional tools.

    Args:
        framework: Framework module name
        name: Agent name
        model_name: LLM model name
        tool_names: List of tool names to create
        **kwargs: Additional agent attributes

    Returns:
        MockAgent instance

    Example:
        ```python
        agent = create_mock_agent(
            framework="crewai",
            name="Researcher",
            tool_names=["search", "scrape"],
        )
        ```
    """
    tools = []
    if tool_names:
        tools = [create_mock_tool(name) for name in tool_names]

    return MockAgent(
        module=framework,
        name=name,
        llm=create_mock_llm(model_name),
        tools=tools,
        **kwargs,
    )


# Assertion helpers for tests


def assert_introspection_has_model(
    result: Any, expected_name: str | None = None
) -> None:
    """
    Assert that introspection result has a baseline model.

    Args:
        result: IntrospectionResult to check
        expected_name: Optional expected model name

    Raises:
        AssertionError: If model is missing or name doesn't match
    """
    assert result.baseline_model is not None, "Expected baseline_model, got None"
    if expected_name:
        assert (
            result.baseline_model.name == expected_name
        ), f"Expected model name '{expected_name}', got '{result.baseline_model.name}'"


def assert_introspection_has_capabilities(
    result: Any,
    expected_count: int | None = None,
    expected_names: list[str] | None = None,
) -> None:
    """
    Assert that introspection result has capabilities.

    Args:
        result: IntrospectionResult to check
        expected_count: Optional expected number of capabilities
        expected_names: Optional list of expected capability names

    Raises:
        AssertionError: If capabilities don't match expectations
    """
    assert result.capabilities, "Expected capabilities, got empty list"

    if expected_count is not None:
        assert (
            len(result.capabilities) == expected_count
        ), f"Expected {expected_count} capabilities, got {len(result.capabilities)}"

    if expected_names:
        actual_names = {c.name for c in result.capabilities}
        missing = set(expected_names) - actual_names
        assert not missing, f"Missing expected capabilities: {missing}"


def assert_introspection_context_has(result: Any, *keys: str) -> None:
    """
    Assert that introspection result context has specific keys.

    Args:
        result: IntrospectionResult to check
        *keys: Context keys that should be present

    Raises:
        AssertionError: If any key is missing
    """
    missing = [k for k in keys if k not in result.context]
    assert not missing, f"Missing context keys: {missing}"
